import time
from typing import List

import einops
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import tqdm
import wandb
from omegaconf import DictConfig

from chicken_and_egg.models.fete import FETE
from chicken_and_egg.trainers.base_trainer import BaseTrainer
from chicken_and_egg.utils.general_utils import to_numpy
from chicken_and_egg.utils.logger import log

sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk")
plt.rc("text", usetex=True)  # camera-ready formatting + latex in plots
plt.rc("font", family="serif")


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
        stage: str = "train",
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
        temp_loss = torch.zeros(self.cfg.num_train_envs, 1).to(self.device)
        episode_return = torch.zeros(self.cfg.num_train_envs, 1).to(self.device)

        # for exploitation, use the context from the exploration policy
        observations_context = context_policy["observations"]
        rewards_context = context_policy["rewards"]
        actions_context = context_policy["actions"]
        mask = context_policy["mask"]
        timesteps = context_policy["timesteps"]
        observations_successor = context_successor["observations"]
        rewards_successor = context_successor["rewards"]
        actions_successor = context_successor["actions"]

        # run a meta_reset here
        temp = self.train_envs.call("meta_reset")
        obs = [o for o, _ in temp]
        obs = torch.stack(obs, dim=0).to(self.device).unsqueeze(1)

        # add the new observations to the context
        observations_context = torch.cat([observations_context, obs], dim=1)[:, 1:]
        observations_successor = torch.cat([observations_successor, obs], dim=1)[:, 1:]

        for ts in range(self.cfg.env.episode_length):
            # [N, T, A], do not update the gradients for the behavior policy
            # these will be updated by copying the weights from the successor policy
            with torch.no_grad():
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
            temp_loss += F.cross_entropy(logits_t, action_t, reduction="none")

            next_state, reward, done, terminal, info = self.train_envs.step(
                to_numpy(action_t)
            )

            # NOTE: only update context if we are exploring or if we are evaluating
            if policy_type == "explore" or stage == "eval":
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
                action_t = einops.repeat(action_t, "b -> b t a", t=1, a=1).detach()
                actions_context = torch.cat([actions_context, action_t], dim=1)[:, 1:]

                timesteps = torch.cat(
                    [timesteps, torch.ones_like(timesteps)[:, :1] * (ts + 1)], dim=1
                )[:, 1:]
                mask = torch.cat([mask, torch.ones_like(mask)[:, :1]], dim=1)[:, 1:]

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
            else:
                reward = torch.from_numpy(reward).to(self.device).unsqueeze(-1).float()

            episode_return += reward

            if done:
                break

        context_policy = {
            "observations": observations_context,
            "rewards": rewards_context,
            "actions": actions_context,
            "mask": mask,
            "timesteps": timesteps,
        }

        context_successor = {
            "observations": observations_successor,
            "rewards": rewards_successor,
            "actions": actions_successor,
        }

        return episode_return, temp_loss, context_policy, context_successor

    def _init_context(self):
        T = self.cfg.env.episode_length * self.cfg.num_episodes

        # keep track of context here for observation, reward and action
        O = self.cfg.env.obs_dim
        A = self.cfg.env.act_dim

        # create context for the behavior explore/exploit policy
        observations_context = torch.zeros(1, T, O).to(self.device)
        rewards_context = torch.zeros(1, T, 1).to(self.device)
        actions_context = torch.zeros(1, T, 1).to(self.device)
        mask = torch.zeros(1, T).to(self.device)
        mask[:, -1] = 1
        timesteps = torch.zeros(1, T).to(self.device).long()

        # create context for the successor explore/exploit policy
        observations_successor = torch.zeros(1, T, O).to(self.device)
        rewards_successor = torch.zeros(1, T, 1).to(self.device)
        actions_successor = torch.zeros(1, T, 1).to(self.device)

        # initialize context with random action and observation
        # obs, info = self.train_envs.reset()
        # obs = torch.from_numpy(obs).to(self.device)
        # observations_context[:, -1] = obs

        context_policy = {
            "observations": observations_context,
            "rewards": rewards_context,
            "actions": actions_context,
            "mask": mask,
            "timesteps": timesteps,
        }
        context_successor = {
            "observations": observations_successor,
            "rewards": rewards_successor,
            "actions": actions_successor,
        }

        return context_policy, context_successor

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        update_time = time.time()
        total_loss = torch.zeros(self.cfg.num_train_envs, 1).to(self.device)
        # best_r = torch.zeros(self.cfg.num_train_envs, 1).to(self.device)
        best_r = torch.tensor(-float("inf")).to(self.device)

        # rollout N episodes
        with torch.amp.autocast("cuda"):
            context_policy, context_successor = self._init_context()
            for ep_idx in range(self.cfg.num_episodes):
                r_explore, l_explore, context_policy, context_successor = (
                    self.rollout_meta_episode(
                        "explore", context_policy, context_successor, stage="train"
                    )
                )
                r_exploit, l_exploit, _, _ = self.rollout_meta_episode(
                    "exploit", context_policy, context_successor, stage="train"
                )

                # exploit episode is 'informative'
                mask = r_exploit >= best_r
                total_loss += l_exploit * mask

                # explore episode is 'maximal'
                mask2 = r_exploit > best_r
                total_loss += l_explore * mask2
                # best_r = r_exploit * mask + best_r * (1 - mask2.int())

                # select the best reward from all the exploit episodes across environments
                r_exploit_ = r_exploit[mask2]
                # handle max of empty tensor
                if r_exploit_.numel() > 0:
                    best_r = r_exploit_.max()
                else:
                    best_r = best_r

                log(f"Epoch: {self.current_epoch}, Best R: {best_r}")

        # average loss over number of environments
        total_loss = total_loss.mean()
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

    def _generate_plots(self, rewards):
        # TODO: pick the first environment for plotting
        # make a plot of rewards over time
        ep_ret = np.cumsum(rewards[0, :, 0], axis=0)
        plt.figure(figsize=(10, 5))
        plt.plot(ep_ret)
        # add vertical line at each episode end
        for i in range(self.cfg.num_episodes):
            plt.axvline(x=i * self.cfg.env.episode_length, color="k", linestyle="--")

        plt.title("Episode Return")
        plt.xlabel("Environment Steps")
        plt.ylabel("Cumulative Reward")
        plt.tight_layout()

        self.log_to_wandb({"ep_ret": wandb.Image(plt)}, prefix="plots/")

    def eval(self):
        log(
            " ======================= Running evaluation episodes ======================= ",
            color="blue",
        )
        self.model.eval()

        num_explore = 1
        num_exploit = self.cfg.num_episodes - num_explore

        context_policy, context_successor = self._init_context()
        with torch.no_grad():
            # we combine the exploit and explore policies for evaluation
            for _ in range(num_explore):
                r_explore, _, context_policy, context_successor = (
                    self.rollout_meta_episode(
                        "explore", context_policy, context_successor, stage="eval"
                    )
                )

            for _ in range(num_exploit):
                r_exploit, _, context_policy, context_successor = (
                    self.rollout_meta_episode(
                        "exploit", context_policy, context_successor, stage="eval"
                    )
                )

            ep_return = r_explore + r_exploit
            mean_ep_return = ep_return.mean().item()
            std_ep_return = ep_return.std().item()

            # generate some visualizations of the return over time
            rewards = context_policy["rewards"]
            rewards = rewards.cpu().numpy()
            actions = context_policy["actions"]
            observations = context_policy["observations"]

            self._generate_plots(rewards)

            eval_metrics = {
                "eval/mean_ep_ret": mean_ep_return,
                "eval/std_ep_ret": std_ep_return,
            }

        return eval_metrics

    def train(self):
        if not self.cfg.skip_first_eval:
            eval_metrics = self.eval()

        for self.current_epoch in tqdm.tqdm(
            range(self.cfg.num_epochs), desc="Training", total=self.cfg.num_epochs
        ):
            train_metrics = self.train_step()

            if self.current_epoch % self.cfg.eval_every == 0:
                eval_metrics = self.eval()

                if self.cfg.use_wandb:
                    wandb.log(eval_metrics)

                log(
                    f"Epoch {self.current_epoch}: Train Loss = {train_metrics['loss']:.4f}, Eval Mean Ep Ret = {eval_metrics['eval/mean_ep_ret']:.4f}, Eval Std Ep Ret = {eval_metrics['eval/std_ep_ret']:.4f}",
                    color="blue",
                )

            # update behavior policy to be same as successor policy every T epochs
            if self.current_epoch % self.cfg.update_behavior_every == 0:
                self.model.update_behavior_policy()

        if self.wandb_run is not None:
            self.wandb_run.finish()
