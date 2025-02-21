import torch


class Bandit:
    def __init__(self, n=10, deterministic=True, noise_scale=0.1):
        self.n = n
        self.arm_means = torch.normal(0, 1, (n,))
        self.deterministic = deterministic
        self.noise_scale = noise_scale

    def gen_arm_means(self):
        return torch.normal(0, 1, (self.n,))

    def reset(self, arm_means=None):
        if arm_means is None:
            arm_means = self.arm_means
        return {"arm_means": arm_means, "reward": 0}

    def meta_reset(self):
        arm_means = self.gen_arm_means()
        return {"arm_means": arm_means, "reward": 0}

    def step(self, state, action):
        if self.deterministic:
            return {
                "arm_means": state["arm_means"],
                "reward": state["arm_means"][action].item(),
            }
        else:
            reward = (
                state["arm_means"][action] + self.noise_scale * torch.normal(0, 1, (1,))
            ).item()
            return {"arm_means": state["arm_means"], "reward": reward}


class MeanBandit:
    def __init__(self, n=10, deterministic=False, noise_scale=0.5, minval=0.5):
        self.n = n
        self.minval = minval
        self.arm_means = self.gen_arm_means()
        self.deterministic = deterministic
        self.noise_scale = noise_scale

    def gen_arm_means(self):
        means = torch.normal(0, 1, (self.n,))
        means[0] = self.minval
        return means

    def reset(self, arm_means=None):
        if arm_means is None:
            arm_means = self.arm_means
        return {"arm_means": arm_means, "reward": 0}

    def meta_reset(self):
        arm_means = self.gen_arm_means()
        return {"arm_means": arm_means, "reward": 0}

    def step(self, state, action):
        if self.deterministic:
            return {
                "arm_means": state["arm_means"],
                "reward": state["arm_means"][action].item(),
            }
        else:
            noise = 0 if action == 0 else self.noise_scale * torch.normal(0, 1, (1,))
            reward = (state["arm_means"][action] + noise).item()
            return {"arm_means": state["arm_means"], "reward": reward}
