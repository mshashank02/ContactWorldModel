#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from custom_envs.dynamic_touch_env import DynamicXMLTouchEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open an on-screen MuJoCo viewer for a generated ShadowHand XML.")
    p.add_argument("--xml-path", required=True, help="Generated env XML to load.")
    p.add_argument("--steps", type=int, default=1000, help="Number of policy steps to display.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-mode", choices=["zero", "random"], default="zero")
    p.add_argument("--target-position", default="random", choices=["random", "ignore"])
    p.add_argument("--target-rotation", default="xyz", choices=["xyz"])
    p.add_argument("--ignore-z-rot", action="store_true")
    p.add_argument("--action-scale", type=float, default=0.6)
    p.add_argument("--action-clip", type=float, default=0.8)
    p.add_argument("--action-smoothing", type=float, default=0.55)
    p.add_argument("--reset-settle-steps", type=int, default=10)
    p.add_argument("--sleep", type=float, default=0.02, help="Seconds to wait after each rendered env step.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.xml_path = os.path.abspath(args.xml_path)
    if not os.path.isfile(args.xml_path):
        raise FileNotFoundError(f"XML not found at {args.xml_path}")
    print(f"[render] xml={args.xml_path}")
    rng = np.random.default_rng(args.seed)
    env = DynamicXMLTouchEnv(
        xml_path=args.xml_path,
        target_position=args.target_position,
        target_rotation=args.target_rotation,
        ignore_z_target_rotation=args.ignore_z_rot,
        render_mode="human",
        action_scale=args.action_scale,
        action_clip=args.action_clip,
        action_smoothing=args.action_smoothing,
        reset_settle_steps=args.reset_settle_steps,
    )
    try:
        env.reset(seed=args.seed)
        for step in range(args.steps):
            if args.action_mode == "zero":
                action = np.zeros(env.action_space.shape, dtype=np.float64)
            else:
                action = rng.uniform(-1.0, 1.0, size=env.action_space.shape)
            _, _, terminated, truncated, info = env.step(action)
            if step % 50 == 0:
                success = info.get("is_success", "n/a") if isinstance(info, dict) else "n/a"
                print(f"[render] step={step} success={success}")
            if terminated or truncated:
                env.reset()
            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
