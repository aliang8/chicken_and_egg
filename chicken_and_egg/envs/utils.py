from chicken_and_egg.envs.bandit import Bandit, MeanBandit


def make_envs(env_name: str, num_envs: int, seed: int):
    from functools import partial

    import gymnasium as gym

    def env_fn(env_idx: int):
        if env_name == "bandit":
            return Bandit(n=10, deterministic=True, noise_scale=0.1)
        elif env_name == "bandit_mean":
            return MeanBandit(n=10, deterministic=False, noise_scale=0.5, minval=0.5)

    envs = [partial(env_fn, env_idx=i) for i in range(num_envs)]
    if num_envs == 1:
        envs = gym.vector.SyncVectorEnv(envs)
    else:
        envs = gym.vector.AsyncVectorEnv(envs)
    return envs
