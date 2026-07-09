#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np


def _finite_array(name: str, arr, limit: float | None = None) -> dict[str, Any]:
    arr = np.asarray(arr)
    finite = bool(np.all(np.isfinite(arr)))
    finite_vals = arr[np.isfinite(arr)]
    max_abs = float(np.max(np.abs(finite_vals))) if finite_vals.size else float("inf")
    ok = finite and (limit is None or max_abs <= limit)
    return {"name": name, "ok": ok, "finite": finite, "max_abs": max_abs, "limit": limit}


def _raise_if_bad(stage: str, checks: list[dict[str, Any]]) -> None:
    bad = [item for item in checks if not item["ok"]]
    if bad:
        detail = "; ".join(
            f"{item['name']} finite={item['finite']} max_abs={item['max_abs']:.6g} limit={item['limit']}"
            for item in bad
        )
        raise FloatingPointError(f"{stage} failed finite/bounds checks: {detail}")


def _check_mujoco_state(stage: str, data, qacc_limit: float, qvel_limit: float) -> None:
    checks = [
        _finite_array("qpos", data.qpos),
        _finite_array("qvel", data.qvel, qvel_limit),
        _finite_array("qacc", data.qacc, qacc_limit),
        _finite_array("ctrl", data.ctrl),
    ]
    _raise_if_bad(stage, checks)


def compile_xml(xml_path: str, qacc_limit: float, qvel_limit: float) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    _check_mujoco_state("compile", data, qacc_limit, qvel_limit)
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "timestep": float(model.opt.timestep),
    }


def passive_rollout(xml_path: str, steps: int, qacc_limit: float, qvel_limit: float) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    if model.nu:
        data.ctrl[:] = 0.0

    last_time = float(data.time)
    max_ncon = 0
    for step in range(int(steps)):
        mujoco.mj_step(model, data)
        if float(data.time) + 1e-12 < last_time:
            raise RuntimeError(f"passive rollout time moved backwards at step {step}")
        last_time = float(data.time)
        max_ncon = max(max_ncon, int(data.ncon))
        _check_mujoco_state(f"passive step {step}", data, qacc_limit, qvel_limit)

    return {"steps": int(steps), "final_time": float(data.time), "max_ncon": max_ncon}


def _check_obs(stage: str, obs) -> None:
    if isinstance(obs, dict):
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                _raise_if_bad(f"{stage}:{key}", [_finite_array(key, value)])
    elif isinstance(obs, np.ndarray):
        _raise_if_bad(stage, [_finite_array("obs", obs)])


def gym_rollout(
    xml_path: str,
    steps: int,
    seed: int,
    target_position: str,
    target_rotation: str,
    ignore_z_rot: bool,
    action_mode: str,
    action_scale: float,
    action_clip: float | None,
    action_smoothing: float,
    reset_settle_steps: int,
    qacc_limit: float,
    qvel_limit: float,
) -> dict[str, Any]:
    from custom_envs.dynamic_touch_env import DynamicXMLTouchEnv

    env = DynamicXMLTouchEnv(
        xml_path=xml_path,
        target_position=target_position,
        target_rotation=target_rotation,
        ignore_z_target_rotation=ignore_z_rot,
        action_scale=action_scale,
        action_clip=action_clip,
        action_smoothing=action_smoothing,
        reset_settle_steps=reset_settle_steps,
    )
    try:
        obs, _ = env.reset(seed=seed)
        _check_obs(f"{action_mode} reset", obs)
        _check_mujoco_state(f"{action_mode} reset", env.data, qacc_limit, qvel_limit)

        rng = np.random.default_rng(seed)
        reward_total = 0.0
        for step in range(int(steps)):
            if action_mode == "zero":
                action = np.zeros(env.action_space.shape, dtype=np.float64)
            else:
                action = rng.uniform(-1.0, 1.0, size=env.action_space.shape)
            obs, reward, terminated, truncated, _ = env.step(action)
            _check_obs(f"{action_mode} step {step}", obs)
            _check_mujoco_state(f"{action_mode} step {step}", env.data, qacc_limit, qvel_limit)
            reward_total += float(reward)
            if terminated or truncated:
                obs, _ = env.reset()
                _check_obs(f"{action_mode} reset-after-done {step}", obs)

        return {"steps": int(steps), "reward_total": reward_total}
    finally:
        env.close()


def training_smoke(
    xml_path: str,
    steps: int,
    seed: int,
    target_position: str,
    target_rotation: str,
    ignore_z_rot: bool,
    action_scale: float,
    action_clip: float | None,
    action_smoothing: float,
    reset_settle_steps: int,
) -> dict[str, Any]:
    if int(steps) <= 0:
        return {"steps": 0, "skipped": True}

    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from sb3_contrib import TQC
    from ShadowHand_TQC import make_env

    smoke_episode_steps = max(10, min(50, int(steps)))
    env = DummyVecEnv(
        [
            make_env(
                xml_path,
                seed,
                0,
                target_position,
                target_rotation,
                ignore_z_rot,
                max_steps=smoke_episode_steps,
                action_scale=action_scale,
                action_clip=action_clip,
                action_smoothing=action_smoothing,
                reset_settle_steps=reset_settle_steps,
            )
        ]
    )
    env = VecNormalize(env, gamma=0.95)
    try:
        model = TQC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={"goal_selection_strategy": "future", "n_sampled_goal": 2},
            learning_starts=smoke_episode_steps + 1,
            train_freq=1,
            gradient_steps=1,
            batch_size=32,
            buffer_size=1000,
            gamma=0.95,
            learning_rate=1e-3,
            policy_kwargs={"net_arch": [64, 64], "n_critics": 1},
            seed=seed,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=int(steps))
        return {"steps": int(steps), "skipped": False}
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate generated deformable XML under MuJoCo and real Gym action rollouts.")
    p.add_argument("--xml-path", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-position", default="random", choices=["random", "ignore"])
    p.add_argument("--target-rotation", default="xyz", choices=["xyz"])
    p.add_argument("--ignore-z-rot", action="store_true")
    p.add_argument("--passive-steps", type=int, default=2000)
    p.add_argument("--env-steps", type=int, default=80)
    p.add_argument("--training-steps", type=int, default=64)
    p.add_argument("--action-scale", type=float, default=0.6)
    p.add_argument("--action-clip", type=float, default=0.8)
    p.add_argument("--action-smoothing", type=float, default=0.55)
    p.add_argument("--reset-settle-steps", type=int, default=10)
    p.add_argument("--qacc-limit", type=float, default=1e9)
    p.add_argument("--qvel-limit", type=float, default=1e6)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    xml_path = os.path.abspath(args.xml_path)
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(xml_path)

    action_clip = args.action_clip if args.action_clip and args.action_clip > 0 else None
    summary: dict[str, Any] = {"xml_path": xml_path, "stages": {}}

    stages = [
        ("compile", lambda: compile_xml(xml_path, args.qacc_limit, args.qvel_limit)),
        ("passive", lambda: passive_rollout(xml_path, args.passive_steps, args.qacc_limit, args.qvel_limit)),
        (
            "gym_zero",
            lambda: gym_rollout(
                xml_path,
                args.env_steps,
                args.seed,
                args.target_position,
                args.target_rotation,
                args.ignore_z_rot,
                "zero",
                args.action_scale,
                action_clip,
                0.0,
                args.reset_settle_steps,
                args.qacc_limit,
                args.qvel_limit,
            ),
        ),
        (
            "gym_random",
            lambda: gym_rollout(
                xml_path,
                args.env_steps,
                args.seed + 1,
                args.target_position,
                args.target_rotation,
                args.ignore_z_rot,
                "random",
                1.0,
                1.0,
                0.0,
                args.reset_settle_steps,
                args.qacc_limit,
                args.qvel_limit,
            ),
        ),
        (
            "gym_random_clipped_smoothed",
            lambda: gym_rollout(
                xml_path,
                args.env_steps,
                args.seed + 2,
                args.target_position,
                args.target_rotation,
                args.ignore_z_rot,
                "random",
                args.action_scale,
                action_clip,
                args.action_smoothing,
                args.reset_settle_steps,
                args.qacc_limit,
                args.qvel_limit,
            ),
        ),
        (
            "training_smoke",
            lambda: training_smoke(
                xml_path,
                args.training_steps,
                args.seed + 3,
                args.target_position,
                args.target_rotation,
                args.ignore_z_rot,
                args.action_scale,
                action_clip,
                args.action_smoothing,
                args.reset_settle_steps,
            ),
        ),
    ]

    for name, fn in stages:
        print(f"[preflight] starting {name}")
        summary["stages"][name] = fn()
        print(f"[preflight] passed {name}: {summary['stages'][name]}")

    print("[preflight] summary " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[preflight] FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
