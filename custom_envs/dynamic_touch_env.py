import numpy as np
import xml.etree.ElementTree as ET
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
        **kwargs,
    ):
        # Gymnasium-Robotics asserts that int(round(1/dt)) equals metadata['render_fps'].
        # For custom XMLs (e.g., timestep=1e-4), align render_fps dynamically.
        n_substeps = int(kwargs.get("n_substeps", 20))
        timestep = _infer_model_timestep(xml_path, default=0.002)
        dt = max(1e-12, timestep * max(1, n_substeps))
        self.metadata = dict(getattr(self, "metadata", {}))
        self.metadata["render_fps"] = int(round(1.0 / dt))

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
            **kwargs,
        )
