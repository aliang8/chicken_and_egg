from typing import List, Tuple

import tqdm
from omegaconf import DictConfig

from chicken_and_egg.models.dream import DREAM
from chicken_and_egg.trainers.base_trainer import BaseTrainer
from chicken_and_egg.utils.data_utils import Transition
from chicken_and_egg.utils.logger import log
from chicken_and_egg.utils.replay_buffer import ReplayBuffer, SequentialReplayBuffer


class DREAMTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.current_epoch = 0

        # Initialize replay buffer for exploration agent
        if self.cfg.buffer.type == "sequential":
            buffer_cls = SequentialReplayBuffer
        else:
            buffer_cls = ReplayBuffer

        self.exploration_rb = buffer_cls(**self.cfg.buffer)
        self.exploitation_rb = buffer_cls(**self.cfg.buffer)

    def setup_model(self):
        model = DREAM(self.cfg.model)
        return model

    def load_checkpoint(self):
        """Load checkpoint using BaseTrainer's functionality"""
        # TODO: Implement this
        pass

    def rollout_trial(self, env, test=False, exploration=False) -> Tuple[List, List]:
        """Runs a single trial following the given policy."""
        trial = []
        renders = []
        obs, info = env.reset()
        hidden_state = None

        if exploration:
            policy = self.model.exploration_policy
        else:
            policy = self.model.exploitation_policy

        while True:
            action, next_hidden_state = policy.select_action(
                obs=obs,
                hidden_state=hidden_state,
                test=test,
            )
            next_obs, reward, done, truncated, info = env.step(action)

            transition = Transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                info=info,
            )
            trial.append(transition)

            obs = next_obs
            hidden_state = next_hidden_state

            if not test:
                # add to replay buffer and perform update
                if exploration:
                    self.exploration_rb.add(transition)
                    exploration_batch = self.exploration_rb.sample(
                        self.cfg.data.batch_size
                    )
                    self.model.update_exploration(exploration_batch)
                else:
                    self.exploitation_rb.add(transition)
                    exploitation_batch = self.exploitation_rb.sample(
                        self.cfg.data.batch_size
                    )
                    self.model.update_exploitation(exploitation_batch)

            if done:
                break

        return trial, renders

    def train_episode(self):
        """Run a single trial in the DREAM algorithm"""
        # Run single trial of exploration
        explore_trial, _ = self.rollout_trial(self.train_envs, exploration=True)

        # Run single trial of exploitation
        exploitation_episode, _ = self.rollout_trial(self.train_envs, exploration=False)

        return metrics

    # def eval(self):
    #     """Run evaluation episodes"""
    #     log("Running evaluation episodes", color="blue")

    #     test_rewards = []
    #     test_exploration_lengths = []

    #     # We don't have task IDs during evaluation
    #     self.trajectory_embedder.use_ids(False)

    #     for test_idx in range(self.cfg.num_eval_episodes):
    #         # Run exploration episode
    #         explore_trial, exploration_renders = self.rollout_trial(
    #             self.eval_envs, self.exploration_policy, test=True
    #         )
    #         test_exploration_lengths.append(len(explore_trial))

    #         # Run exploitation episode
    #         episode, renders = self.rollout_trial(
    #             self.eval_envs, self.exploit_policy, test=True
    #         )
    #         test_rewards.append(sum(exp.reward for exp in episode))

    #     eval_metrics = {
    #         "eval/mean_reward": np.mean(test_rewards),
    #         "eval/std_reward": np.std(test_rewards),
    #         "eval/mean_exploration_length": np.mean(test_exploration_lengths),
    #     }

    #     self.log_to_wandb(eval_metrics)
    #     return eval_metrics

    def train(self):
        """Main training loop following Algorithm 2 from the paper"""
        # Load checkpoint if specified
        if self.cfg.load_from_ckpt:
            self.load_checkpoint()
        elif not self.cfg.skip_first_eval:
            self.eval()

        for self.current_epoch in tqdm.tqdm(
            range(self.current_epoch, self.cfg.num_epochs), desc="Training"
        ):
            train_metrics = self.train_episode()

            if self.current_epoch % self.cfg.eval_every == 0:
                eval_metrics = self.eval()
                self.save_model(self.save_dict, eval_metrics, self.current_epoch)

                log(
                    f"Epoch {self.current_epoch}: "
                    f"Train Reward = {train_metrics['reward/train']:.4f}, "
                    f"Eval Reward = {eval_metrics['eval/mean_reward']:.4f}",
                    color="blue",
                )

        if self.wandb_run is not None:
            self.wandb_run.finish()
