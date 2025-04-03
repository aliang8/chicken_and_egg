import gymnasium as gym
import numpy as np

from chicken_and_egg.utils.logger import log


class Bandit(gym.Env):
    def __init__(self, n=10, deterministic=True, noise_scale=0.1, **kwargs):
        super().__init__()
        self.n = n
        self.arm_means = np.random.randn(n)
        self.deterministic = deterministic
        self.noise_scale = noise_scale
        self.current_state = None
        log(f"Best reward: {self._calculate_max_reward()}", color="yellow")

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.n)

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(self.n,))

    def gen_arm_means(self):
        return np.random.randn(self.n)

    def reset(self, arm_means=None, sample_arm_means=False, **kwargs):
        if arm_means is None:
            arm_means = self.arm_means
        self.current_state = arm_means

        if sample_arm_means:
            arm_means = self.gen_arm_means()
            self.current_state = arm_means

        # for debugging
        log(f"Arm means: {arm_means}", color="yellow")
        return arm_means, {"reward": 0}

    def step(self, action):
        if self.deterministic:
            return (
                self.current_state,
                self.current_state[action].item(),
                False,
                False,
                {"reward": self.current_state[action].item()},
            )
        else:
            reward = (
                self.current_state[action] + self.noise_scale * np.random.randn()
            ).item()
            return self.current_state, reward, False, False, {"reward": reward}

    def _calculate_max_reward(self):
        # best reward is the max of the arm means
        return self.arm_means.max().item()


class MeanBandit(Bandit):
    def __init__(self, n=10, deterministic=False, noise_scale=0.5, minval=0.5):
        super().__init__(n=n, deterministic=deterministic, noise_scale=noise_scale)
        self.minval = minval

    def gen_arm_means(self):
        means = np.random.randn(self.n)
        means[0] = self.minval
        return means

    def step(self, action):
        if self.deterministic:
            return super().step(action)
        else:
            noise = 0 if action == 0 else self.noise_scale * np.random.randn()
            reward = (self.current_state[action] + noise).item()
            return self.current_state, reward, False, False, {"reward": reward}
