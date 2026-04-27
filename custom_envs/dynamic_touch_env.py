import numpy as np
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
        **kwargs,
    ):
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
            **kwargs,
        )

    def reset(self, *, seed=None, options=None):
        try:
            obs, info = super().reset(seed=seed, options=options)
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
