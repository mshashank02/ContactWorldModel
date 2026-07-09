import numpy as np
import os
import xml.etree.ElementTree as ET
import traceback
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.shadow_dexterous_hand import MujocoManipulateTouchSensorsEnv


def _infer_model_timestep(xml_path: str, default: float = 0.002) -> float:
    try:
        root = ET.parse(xml_path).getroot()
        opt = root.find("option")
        if opt is None:
            return default
        ts = opt.get("timestep")
        if ts is None:
            return default
        val = float(ts)
        return val if val > 0 else default
    except Exception:
        return default


class DynamicXMLTouchEnv(MujocoManipulateTouchSensorsEnv, EzPickle):
    def __init__(
        self,
        xml_path,
        target_position="random",
        target_rotation="xyz",
        ignore_z_target_rotation=False,
        touch_get_obs="sensordata",
        reward_type="sparse",
        debug_goal_print=False,
        debug_goal_print_every=1,
        action_scale=1.0,
        action_clip=None,
        action_smoothing=0.0,
        reset_settle_steps=0,
        **kwargs,
    ):
        xml_path = os.path.abspath(xml_path)
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"XML not found at {xml_path}")

        # Gymnasium-Robotics asserts that int(round(1/dt)) equals metadata['render_fps'].
        # For custom XMLs (e.g., timestep=1e-4), align render_fps dynamically.
        n_substeps = int(kwargs.get("n_substeps", 20))
        timestep = _infer_model_timestep(xml_path, default=0.002)
        dt = max(1e-12, timestep * max(1, n_substeps))
        self.metadata = dict(getattr(self, "metadata", {}))
        self.metadata["render_fps"] = int(round(1.0 / dt))
        self.debug_goal_print = bool(debug_goal_print)
        self.debug_goal_print_every = max(1, int(debug_goal_print_every))
        self._debug_step_idx = 0
        self.action_scale = float(action_scale)
        self.action_clip = None if action_clip is None else float(action_clip)
        if self.action_clip is not None and self.action_clip <= 0:
            self.action_clip = None
        self.action_smoothing = float(np.clip(action_smoothing, 0.0, 0.98))
        self.reset_settle_steps = max(0, int(reset_settle_steps))
        self._last_action = None

        MujocoManipulateTouchSensorsEnv.__init__(
            self,
            model_path=xml_path,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            ignore_z_target_rotation=ignore_z_target_rotation,
            **kwargs,
        )
        EzPickle.__init__(
            self,
            xml_path,
            target_position,
            target_rotation,
            ignore_z_target_rotation,
            touch_get_obs,
            reward_type,
            debug_goal_print,
            debug_goal_print_every,
            action_scale,
            action_clip,
            action_smoothing,
            reset_settle_steps,
            **kwargs,
        )

    def _finite_state_summary(self):
        parts = []
        for name in ("qpos", "qvel", "qacc", "ctrl"):
            arr = getattr(getattr(self, "data", None), name, None)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.size == 0:
                continue
            finite = np.isfinite(arr)
            max_abs = np.max(np.abs(arr[finite])) if np.any(finite) else np.inf
            parts.append(f"{name}:finite={bool(np.all(finite))},max_abs={max_abs:.6g}")
        return "; ".join(parts)

    def _assert_finite_state(self, where):
        for name in ("qpos", "qvel", "qacc", "ctrl"):
            arr = getattr(getattr(self, "data", None), name, None)
            if arr is not None and not np.all(np.isfinite(arr)):
                raise FloatingPointError(f"Non-finite MuJoCo {name} after {where}; {self._finite_state_summary()}")

    def _condition_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        if self.action_scale != 1.0:
            action = action * self.action_scale
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)
        if self.action_smoothing > 0.0:
            if self._last_action is None or self._last_action.shape != action.shape:
                self._last_action = np.zeros_like(action)
            alpha = self.action_smoothing
            action = alpha * self._last_action + (1.0 - alpha) * action
            if self.action_clip is not None:
                action = np.clip(action, -self.action_clip, self.action_clip)
        self._last_action = np.array(action, copy=True)
        return action

    def _set_action(self, action):
        super()._set_action(self._condition_action(action))

    def _settle_after_reset(self):
        if self.reset_settle_steps <= 0:
            return
        zero = np.zeros(self.action_space.shape, dtype=np.float64)
        for _ in range(self.reset_settle_steps):
            self._set_action(zero)
            if hasattr(self, "do_simulation"):
                self.do_simulation(self.data.ctrl, self.n_substeps)
            else:
                self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
            self._assert_finite_state("reset settling")

    def reset(self, *, seed=None, options=None):
        try:
            obs, info = super().reset(seed=seed, options=options)
            self._last_action = np.zeros(self.action_space.shape, dtype=np.float64)
            self._settle_after_reset()
            self._assert_finite_state("reset")
            obs = self._get_obs()
        except Exception as e:
            if self.debug_goal_print:
                print(f"[goal_debug][mujoco_exception][reset] {type(e).__name__}: {e}")
                traceback.print_exc()
            raise
        self._debug_step_idx = 0
        return obs, info

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = super().step(action)
            self._assert_finite_state(f"step={self._debug_step_idx + 1}")
        except Exception as e:
            if self.debug_goal_print:
                print(
                    f"[goal_debug][mujoco_exception][step={self._debug_step_idx + 1}] "
                    f"{type(e).__name__}: {e}"
                )
                traceback.print_exc()
            raise
        self._debug_step_idx += 1
        if self.debug_goal_print and (self._debug_step_idx % self.debug_goal_print_every == 0):
            achieved = np.array2string(
                obs["achieved_goal"], precision=5, suppress_small=True, max_line_width=200
            )
            desired = np.array2string(
                obs["desired_goal"], precision=5, suppress_small=True, max_line_width=200
            )
            print(
                f"[goal_debug] step={self._debug_step_idx} reward={float(reward):.5f} "
                f"terminated={terminated} truncated={truncated}"
            )
            print(f"[goal_debug] achieved_goal={achieved}")
            print(f"[goal_debug] desired_goal ={desired}")
        return obs, reward, terminated, truncated, info
