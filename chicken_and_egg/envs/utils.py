from typing import Dict

from chicken_and_egg.envs.bandit import Bandit, MeanBandit
from chicken_and_egg.envs.darkroom import DarkRoom


def make_envs(env_name: str, num_envs: int, seed: int, env_kwargs: Dict = None):
    from functools import partial

    import gymnasium as gym

    def env_fn(env_idx: int):
        if env_name == "bandit_reg":
            env_cls = Bandit
        elif env_name == "bandit_mean":
            env_cls = MeanBandit
        elif env_name == "darkroom":
            env_cls = DarkRoom
        else:
            raise ValueError(f"Unknown environment: {env_name}")

        env = env_cls(**env_kwargs)
        return env

    envs = [partial(env_fn, env_idx=i) for i in range(num_envs)]
    if num_envs == 1:
        envs = gym.vector.SyncVectorEnv(envs)
    else:
        envs = gym.vector.AsyncVectorEnv(envs)
    return envs
