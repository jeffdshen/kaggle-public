from .utils import ArmHelper, ARMS

from gym.envs.registration import register

register(
    id="Santa2022Game-v0",
    entry_point="santa2022.game:SantaGameEnv",
    max_episode_steps=10000,
)
