from dataclasses import fields
from typing import List, Optional, Tuple

import numpy as np
import torch
import tqdm
from omegaconf import DictConfig

from chicken_and_egg.models.dream import DREAM
from chicken_and_egg.trainers.base_trainer import BaseTrainer
from chicken_and_egg.utils.data_utils import Transition
from chicken_and_egg.utils.replay_buffer import ReplayBuffer, SequentialReplayBuffer


def pad(lol: List[List[Transition]]) -> Tuple[List[Transition], torch.Tensor]:
    max_len = max(len(ls) for ls in lol)
    padded_lol = []
    mask = torch.zeros(len(lol), max_len, dtype=torch.bool)
    for i, ls in enumerate(lol):
        padded_ls = ls + [ls[-1]] * (max_len - len(ls))
        padded_lol.append(padded_ls)
        mask[i, : len(ls)] = True
    return padded_lol, mask


def convert_list_of_lists(
    lol: List[List[Transition]], device: torch.device
) -> Tuple[List[Transition], torch.Tensor]:
    padded_lol, mask = pad(lol)

    field_names = [f.name for f in fields(Transition)]
    transitions = Transition(
        **{
            k: torch.stack(
                [
                    torch.tensor([getattr(t, k) for t in ls], device=device)
                    for ls in padded_lol
                ]
            ).float()
            for k in field_names
            if k not in ["info", "hidden_state", "trajectory"]
        }
    )
    trajectory = [[getattr(t, "trajectory") for t in ls] for ls in padded_lol]
    transitions.trajectory = trajectory
    return transitions, mask.to(device)


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

        # Set up optimizer and scheduler for exploration and exploitation policies
        self.exploration_optimizer = self.get_optimizer(
            self.model.exploration_policy.parameters(), self.cfg.optimizer
        )
        self.exploitation_optimizer = self.get_optimizer(
            self.model.exploit_policy.parameters(), self.cfg.optimizer
        )
        self.exploration_scheduler = self.get_scheduler(
            self.exploration_optimizer, self.cfg.lr_scheduler
        )
        self.exploitation_scheduler = self.get_scheduler(
            self.exploitation_optimizer, self.cfg.lr_scheduler
        )

    def setup_model(self):
        model = DREAM(self.cfg.model)
        return model

    def update(self, batch: List[List[Transition]], policy_name: str):
        """Update exploration policy using batch of transitions"""

        # convert list of transitions to Transition object
        # obs - [B, T, N, D]
        # action - [B, T, N]
        # reward - [B, T, N]
        # next_obs - [B, T, N, D]
        # done - [B, T, N]
        experiences, mask = convert_list_of_lists(batch, self.device)

        # compute loss
        if policy_name == "exploration":
            loss = self.model.exploration_policy._compute_loss(
                experiences, mask, relabel_rewards=True
            )
        else:
            loss = self.model.exploit_policy._compute_loss(experiences, mask)

        # backpropagate
        if policy_name == "exploration":
            self.exploration_optimizer.zero_grad()
            loss.backward()
            self.exploration_optimizer.step()
            self.exploration_scheduler.step()
        else:
            self.exploitation_optimizer.zero_grad()
            loss.backward()
            self.exploitation_optimizer.step()
            self.exploitation_scheduler.step()
        return loss

    def load_checkpoint(self):
        """Load checkpoint using BaseTrainer's functionality"""
        # TODO: Implement this
        pass

    def rollout_trial(
        self,
        env,
        test=False,
        exploration=False,
        context: Optional[List[Transition]] = None,
    ) -> Tuple[List, List]:
        """Runs a single trial following the given policy."""
        trial = []
        renders = []
        obs, info = env.reset()
        hidden_state = None
        timestep = 0

        if exploration:
            policy = self.model.exploration_policy
        else:
            policy = self.model.exploit_policy

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
                env_id=[[1]],
                index=timestep,
                trajectory=context,
            )
            trial.append(transition)

            obs = next_obs
            hidden_state = next_hidden_state
            timestep += 1

            if not test:
                if exploration is False:
                    # add to replay buffer and perform update
                    self.exploitation_rb.add(transition)
                    # start updating after min buffer size is hit
                    if self.exploitation_rb.size() >= self.cfg.start_training_at:
                        exploitation_batch = self.exploitation_rb.sample(
                            self.cfg.data.batch_size
                        )
                        exploitation_loss = self.update(
                            exploitation_batch, "exploitation"
                        )

            if done:
                break

        return trial, renders

    def train_episode(self):
        """Run a single trial in the DREAM algorithm"""
        # Run single trial of exploration
        explore_trial, _ = self.rollout_trial(self.train_envs, exploration=True)
        # print(f"Explore trial length: {len(explore_trial)}")

        # Postprocess exploration trial to get trajectories
        for transition in explore_trial:
            trajectory = Transition(
                obs=np.array([t.obs for t in explore_trial]),
                action=np.array([t.action for t in explore_trial]),
                reward=np.array([t.reward for t in explore_trial]),
                next_obs=np.array([t.next_obs for t in explore_trial]),
                done=np.array([t.done for t in explore_trial]),
                info=np.array([t.info for t in explore_trial]),
                hidden_state=np.array([t.hidden_state for t in explore_trial]),
            )
            transition.trajectory = trajectory

            # Perform update here
            self.exploration_rb.add(transition)
            # start updating after min buffer size is hit
            if self.exploration_rb.size() >= self.cfg.start_training_at:
                exploration_batch = self.exploration_rb.sample(self.cfg.data.batch_size)
                exploration_loss = self.update(exploration_batch, "exploration")

        # Run single trial of exploitation
        exploitation_episode, _ = self.rollout_trial(
            self.train_envs, exploration=False, context=explore_trial
        )
        # print(f"Exploitation episode length: {len(exploitation_episode)}")
        return

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

            # if self.current_epoch % self.cfg.eval_every == 0:
            #     eval_metrics = self.eval()
            #     self.save_model(self.save_dict, eval_metrics, self.current_epoch)

            #     log(
            #         f"Epoch {self.current_epoch}: "
            #         f"Train Reward = {train_metrics['reward/train']:.4f}, "
            #         f"Eval Reward = {eval_metrics['eval/mean_reward']:.4f}",
            #         color="blue",
            #     )

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def setup_optimizer_and_scheduler(self):
        return None, None
