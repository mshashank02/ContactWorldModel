from gymnasium.envs.registration import register

register(
    id="HandManipulateBlock_ForwardFaceTouchSensors-v1",
    entry_point="custom_envs.hand_block_forward_face_env:HandBlockForwardFaceTouchEnv",
    max_episode_steps=50,
)
