import collections

import tqdm
from omegaconf import DictConfig

from chicken_and_egg.trainers.base_trainer import BaseTrainer


class DREAMTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def setup_model(self):
        model = DREAM(self.cfg.model)
        return model

    def eval(self):
        pass

    def train(self):
        episode_lengths = collections.deque(maxlen=200)
        total_steps = 0
        exploration_steps = 0
        instruction_steps = 0

        for episode_num in tqdm.tqdm(range(1000000)):
            exploration_episode, _ = self.rollout_meta_episode()

            for index, exp in enumerate(episode):
                self.model.update(relabel.TrajectoryExperience(exp, episode, index))

            total_steps += len(episode)
            episode_lengths.append(len(episode))
            rewards.append(sum(exp.reward for exp in episode))
