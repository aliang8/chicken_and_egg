import gymnasium as gym
import numpy as np


# Non-deceptive version of darkroom
class DarkRoom(gym.Env):
    def __init__(
        self,
        w,
        h,
        num_treasures=1,
        rand_start=False,
        hard_reward=False,
        minval=0,
        maxval=3,
    ):
        self.rand_start = rand_start
        self.hard_reward = hard_reward
        self.w = w
        self.h = h
        self.num_treasures = num_treasures
        self.minval = minval
        self.maxval = maxval

        self.current_state = None
        # this keeps track of which traps and rewards have been visited
        self.visited = np.zeros((num_treasures,))

    @property
    def action_space(self):
        return gym.spaces.Discrete(5)

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=np.array([0, 0]), high=np.array([self.w, self.h]), dtype=np.int64
        )

    def step(self, action: int):
        # action in 0, 1, 2, 3, 4
        # 0 no-op
        # 1 up, 2 right, 3 down, 4 left
        curr_x, curr_y = self.current_state
        ax = np.clip(curr_x + (action == 2) - (action == 4), 0, self.w - 1)
        ay = np.clip(curr_y + (action == 1) - (action == 3), 0, self.h - 1)

        # check if the agent has visited the treasure or trap
        visited = (ax == self.rx) & (ay == self.ry)
        # reward is the sum of the rewards for the treasures and traps
        reward = np.sum(self.rr[visited])

        obs = self.get_obs(ax, ay)
        self.current_state = (ax, ay)
        return (obs, reward, False, False, self.get_info())

    def reset(self, seed: int = None, **kwargs):
        if seed is not None:
            np.random.seed(seed)

        if self.rand_start:  # start the agent at a random location
            ax = np.random.randint(0, self.w)
            ay = np.random.randint(0, self.h)
        else:  # start the agent at the center of the room
            ax = self.w // 2
            ay = self.h // 2

        # these are the positions of the treasures and traps
        self.rx = np.random.randint(0, self.w, (self.num_treasures,))
        self.ry = np.random.randint(0, self.h, (self.num_treasures,))
        # these are the rewards for the treasures and traps
        self.rr = np.random.uniform(self.minval, self.maxval, (self.num_treasures,))

        self.current_state = (ax, ay)

        return self.get_obs(ax, ay), self.get_info()

    def get_info(self):
        return {
            "rx": self.rx,
            "ry": self.ry,
            "rr": self.rr,
            "w": self.w,
            "h": self.h,
            "visited": self.visited,
        }

    def get_obs(self, ax, ay):
        return np.array([ax, ay])
