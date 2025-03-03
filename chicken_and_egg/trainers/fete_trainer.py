import time
from typing import List

import einops
import torch
import torch.nn.functional as F
import tqdm
import wandb
from omegaconf import DictConfig

from chicken_and_egg.models.fete import FETE
from chicken_and_egg.trainers.base_trainer import BaseTrainer
from chicken_and_egg.utils.general_utils import to_numpy
from chicken_and_egg.utils.logger import log


class FETETrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.current_epoch = 0

    def setup_model(self):
        model = FETE(self.cfg.model)
        return model

    def rollout_meta_episode(
        self,
        policy_type: str,
        context_policy: List[torch.Tensor] = None,
        context_successor: List[torch.Tensor] = None,
    ):
        """
        Rollout the exploration behavior and successor policies.
        Returns the context for the exploitation policy.

        NOTE: the exploration policy provides the context for both policies

        Returns:
            episode_return: float
            temp_loss: float
            context_policy: [observations, rewards, actions]
            context_successor: [observations, rewards, actions]
        """
        temp_loss, episode_return = 0, 0

        if policy_type == "explore":
            # keep track of context here for observation, reward and action
            T = self.cfg.env.episode_length * self.cfg.env.num_meta_episodes
            O = self.cfg.env.obs_dim
            A = self.cfg.env.act_dim

            # create context for the behavior explore/exploit policy
            observations_context = torch.zeros(1, T, O).to(self.device)
            rewards_context = torch.zeros(1, T, 1).to(self.device)
            actions_context = torch.zeros(1, T, 1).to(self.device)
            timesteps = torch.arange(T).unsqueeze(0).to(self.device)
            mask = torch.zeros(1, T).to(self.device)
            mask[-1] = 1

            # create context for the successor explore/exploit policy
            observations_successor = torch.zeros(1, T, O).to(self.device)
            rewards_successor = torch.zeros(1, T, 1).to(self.device)
            actions_successor = torch.zeros(1, T, 1).to(self.device)

            # initialize context with random action and observation
            obs, info = self.train_envs.reset()
            obs = torch.from_numpy(obs).to(self.device)
            observations_context[:, -1] = obs
        else:
            # for exploitation, use the context from the exploration policy
            observations_context, rewards_context, actions_context = context_policy
            observations_successor, rewards_successor, actions_successor = (
                context_successor
            )

        for ts in range(self.cfg.env.episode_length):
            # [N, T, A]
            behavior_logits = self.model(
                observations=observations_context,
                actions=actions_context,
                rewards=rewards_context,
                timesteps=timesteps,
                attention_mask=mask,
                policy_type=f"{policy_type}_behavior",
            )

            # [N, T, A]
            successor_logits = self.model(
                observations=observations_successor,
                actions=actions_successor,
                rewards=rewards_successor,
                timesteps=timesteps,
                attention_mask=mask,
                policy_type=f"{policy_type}_successor",
            )

            # compute hadamard product of logits
            logits = behavior_logits * successor_logits

            # sample action from logits
            action = torch.argmax(logits, dim=-1)

            # compute loss for current timestep
            logits_t = logits[:, -1]
            action_t = action[:, -1]

            # cross entropy loss
            temp_loss += F.cross_entropy(logits_t, action_t)

            next_state, reward, done, terminal, info = self.train_envs.step(
                to_numpy(action_t)
            )

            # NOTE: only update context if we are exploring
            if policy_type == "explore":
                next_state = torch.from_numpy(next_state).to(self.device)
                action_t = action_t.float()
                reward = torch.from_numpy(reward).to(self.device).unsqueeze(-1).float()

                # update context by appending new state, reward and action
                observations_context = torch.cat(
                    [observations_context, next_state.unsqueeze(1)], dim=1
                )[:, 1:]
                rewards_context = torch.cat(
                    [rewards_context, reward.unsqueeze(1)], dim=1
                )[:, 1:]
                action_t = einops.repeat(action_t, "b -> b t a", t=1, a=1)
                actions_context = torch.cat([actions_context, action_t], dim=1)[:, 1:]

                # update successor context by appending new state, reward and action
                observations_successor = torch.cat(
                    [observations_successor, next_state.unsqueeze(1)], dim=1
                )[:, 1:]
                rewards_successor = torch.cat(
                    [rewards_successor, reward.unsqueeze(1)], dim=1
                )[:, 1:]
                actions_successor = torch.cat([actions_successor, action_t], dim=1)[
                    :, 1:
                ]

            episode_return += reward

            if done:
                break

        context_policy = [observations_context, rewards_context, actions_context]
        context_successor = [
            observations_successor,
            rewards_successor,
            actions_successor,
        ]

        if policy_type == "explore":
            return episode_return, temp_loss, context_policy, context_successor
        else:
            return episode_return, temp_loss, None, None

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        update_time = time.time()
        total_loss = 0.0
        best_r = 0.0  # TODO: set this as env parameter

        # rollout N episodes
        for ep_idx in range(self.cfg.env.num_meta_episodes):
            with torch.amp.autocast("cuda"):
                r_explore, l_explore, context_policy, context_successor = (
                    self.rollout_meta_episode("explore")
                )
                r_exploit, l_exploit, _, _ = self.rollout_meta_episode(
                    "exploit", context_policy, context_successor
                )

                if r_exploit > best_r:
                    total_loss += l_exploit
                if r_explore > best_r:
                    total_loss += l_explore
                    best_r = r_exploit

        self.scaler.scale(total_loss).backward()
        # Unscale gradients to prepare for gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.cfg.clip_grad_norm
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        metrics = {}
        metrics["time/update"] = time.time() - update_time
        metrics["lr"] = self.scheduler.get_last_lr()[0]

        train_metrics = {
            "loss": total_loss.item(),
            **metrics,
        }

        self.log_to_wandb(train_metrics, prefix="train/")
        self.log_to_wandb({"_update": self.current_epoch}, prefix="step/")

        return train_metrics

    def eval_step(self, batch_size: int):
        self.model.eval()
        with torch.no_grad():
            states = self.batch_mset(batch_size)
            actions = torch.randint(
                0, self.cfg.env.act_dim, (batch_size, self.cfg.env.seq_len - 1)
            ).to(self.device)
            rewards = self.reward_sequence(states, actions)
            running_max = self.max_in_seq(rewards)
            _, loss = self.exploit(
                states, actions, rewards, running_max, epsilon=self.cfg.train.epsilon
            )
            return loss.item()

    def train(self):
        for self.current_epoch in tqdm.tqdm(
            range(self.cfg.num_epochs), desc="Training", total=self.cfg.num_epochs
        ):
            train_metrics = self.train_step()

            if self.current_epoch % self.cfg.eval_every == 0:
                eval_metrics = self.eval_step()

                if self.cfg.use_wandb:
                    wandb.log(eval_metrics)

                log(
                    f"Epoch {self.current_epoch}: Train Loss = {train_metrics['loss']:.4f}, Eval Loss = {eval_metrics['loss']:.4f}"
                )

            # update behavior policy to be same as successor policy every T epochs
            if self.current_epoch % self.cfg.update_behavior_every == 0:
                self.model.update_behavior_policy()

        if self.wandb_run is not None:
            self.wandb_run.finish()
