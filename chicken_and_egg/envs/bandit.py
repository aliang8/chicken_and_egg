import gymnasium as gym
import torch


class Bandit(gym.Env):
    def __init__(self, n=10, deterministic=True, noise_scale=0.1):
        super().__init__()
        self.n = n
        self.arm_means = torch.normal(0, 1, (n,))
        self.deterministic = deterministic
        self.noise_scale = noise_scale
        self.current_state = None

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.n)

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(self.n,))

    def gen_arm_means(self):
        return torch.normal(0, 1, (self.n,))

    def reset(self, seed=None, arm_means=None, **kwargs):
        if arm_means is None:
            arm_means = self.arm_means
        self.current_state = arm_means
        return arm_means, {"reward": 0}

    def meta_reset(self):
        arm_means = self.gen_arm_means()
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
                self.current_state[action] + self.noise_scale * torch.normal(0, 1, (1,))
            ).item()
            return self.current_state, reward, False, False, {"reward": reward}

    def _calculate_max_reward(self):
        return (
            torch.normal(0, 1, size=(10000, self.cfg.env.act_dim))
            .max(dim=1)[0]
            .mean()
            .item()
        )


class MeanBandit(Bandit):
    def __init__(self, n=10, deterministic=False, noise_scale=0.5, minval=0.5):
        super().__init__(n=n, deterministic=deterministic, noise_scale=noise_scale)
        self.n = n
        self.minval = minval
        self.arm_means = self.gen_arm_means()
        self.deterministic = deterministic
        self.noise_scale = noise_scale

    def gen_arm_means(self):
        means = torch.normal(0, 1, (self.n,))
        means[0] = self.minval
        return means

    def reset(self, seed=None, arm_means=None, **kwargs):
        if arm_means is None:
            arm_means = self.arm_means
        self.current_state = arm_means
        return arm_means, {"reward": 0}

    def meta_reset(self):
        arm_means = self.gen_arm_means()
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
            noise = 0 if action == 0 else self.noise_scale * torch.normal(0, 1, (1,))
            reward = (self.current_state[action] + noise).item()
            return self.current_state, reward, False, False, {"reward": reward}

    def _calculate_max_reward(self):
        vals = torch.normal(0, 1, size=(10000, self.cfg.env.act_dim))
        vals[:, 0] = 0.5
        return vals.max(dim=1)[0].mean().item()
