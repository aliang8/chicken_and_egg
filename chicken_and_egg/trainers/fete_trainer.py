import time
from typing import Dict

import einops
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
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

    def rollout_trial(
        self,
        env: gym.Env,
        policy_type: str,
        context: Dict[str, torch.Tensor] = None,
        apply_meta_reset: bool = True,
        trial_id: int = None,
    ):
        """
        Rollout a trial for the GPT-2 based transformer policy.

        Returns:
            trial_return: float
            action_loss: float
            context: Dict[str, torch.Tensor], [N, T, ...]
        """
        action_loss = torch.zeros(env.num_envs, 1).to(self.device)
        trial_return = torch.zeros(env.num_envs, 1).to(self.device)

        # for exploitation, use the context from the exploration policy
        observations_context = context["observations"]
        rewards_context = context["rewards"]
        actions_context = context["actions"]
        mask = context["mask"]
        timesteps = context["timesteps"]
        trial_ids = context["trial_ids"]
        infos = context["infos"]

        if apply_meta_reset:
            obs, info = env.reset()
            obs = torch.from_numpy(obs).to(self.device).unsqueeze(1)

            # add the new observations to the context
            observations_context = torch.cat([observations_context, obs], dim=1)[:, 1:]

            # roll mask
            mask = torch.roll(mask, shifts=-1, dims=1)
            # roll all of the others
            actions_context = torch.roll(actions_context, shifts=-1, dims=1)
            rewards_context = torch.roll(rewards_context, shifts=-1, dims=1)
            # set the last mask to 1
            mask[:, -1] = 1

            # roll trial_id
            trial_ids = torch.roll(trial_ids, shifts=-1, dims=1)
            trial_ids[:, -1] = trial_id
            infos.append(info)

        for ts in range(self.cfg.env.timesteps_per_trial):
            # [N, T, A], do not update the gradients for the behavior policy
            # these will be updated by copying the weights from the successor policy
            policy_kwargs = {
                "observations": observations_context,
                "actions": actions_context,
                "rewards": rewards_context,
                "timesteps": timesteps,
                "attention_mask": mask,
                "trial_ids": trial_ids,
            }

            # don't update the behavior policy, see paper
            with torch.no_grad():
                behavior_logits = self.model(
                    **policy_kwargs, policy_type=f"{policy_type}_behavior"
                )

            # [N, T, A]
            successor_logits = self.model(
                **policy_kwargs, policy_type=f"{policy_type}_successor"
            )

            # compute hadamard product of logits
            logits = behavior_logits * successor_logits

            # sample action from logits
            action = torch.argmax(logits, dim=-1)

            # compute loss for current timestep
            logits_t = logits[:, -1]
            action_t = action[:, -1]

            # cross entropy loss
            action_loss += F.cross_entropy(logits_t, action_t, reduction="mean")

            next_state, reward, done, terminal, info = env.step(to_numpy(action_t))

            next_state = torch.from_numpy(next_state).to(self.device)
            action_t = action_t.float()
            reward = torch.from_numpy(reward).to(self.device).unsqueeze(-1).float()

            # update context by appending new state, reward and action
            # if its the last timestep, don't append the next state
            if ts < self.cfg.env.timesteps_per_trial - 1:
                observations_context = torch.cat(
                    [observations_context, next_state.unsqueeze(1)], dim=1
                )[:, 1:]

            # apply roll here is to account for the
            # first observation which doesn't have an associated reward / action
            rewards_context[:, -1] = reward
            rewards_context = torch.roll(rewards_context, shifts=-1, dims=1)
            action_t = einops.repeat(action_t, "b -> b t a", t=1, a=1).detach()
            actions_context[:, -2:-1] = action_t
            actions_context = torch.roll(actions_context, shifts=-1, dims=1)

            timesteps = torch.cat(
                [timesteps, torch.ones_like(timesteps)[:, :1] * (ts + 1)], dim=1
            )[:, 1:]
            mask = torch.cat([mask, torch.ones_like(mask)[:, :1]], dim=1)[:, 1:]
            trial_return += reward

            # add the trial id to the context
            trial_id_tensor = torch.tensor(trial_id).long()
            trial_id_tensor = einops.repeat(
                trial_id_tensor, " -> b t", b=env.num_envs, t=1
            ).to(self.device)
            trial_ids = torch.cat([trial_ids, trial_id_tensor], dim=1)[:, 1:]
            infos.append(info)

            # assumes all envs finish at the same time
            if done.all():
                break

        new_context = {
            "observations": observations_context,
            "rewards": rewards_context,
            "actions": actions_context,
            "mask": mask,
            "timesteps": timesteps,
            "trial_ids": trial_ids,
            "infos": infos,
        }
        return trial_return, action_loss, new_context

    def _init_context(self, batch_size: int = 1):
        T = self.cfg.env.timesteps_per_trial * self.cfg.num_trials
        # add some dummy timesteps for the initial obs
        # T += self.cfg.num_trials

        # keep track of context here for observation, reward and action
        O = self.cfg.env.obs_dim
        A = self.cfg.env.act_dim

        # create context for the behavior explore/exploit policy
        observations_context = torch.zeros(batch_size, T, O).to(self.device)
        rewards_context = torch.zeros(batch_size, T, 1).to(self.device)
        actions_context = torch.zeros(batch_size, T, 1).to(self.device)
        mask = torch.zeros(batch_size, T).to(self.device)
        timesteps = torch.zeros(batch_size, T).to(self.device).long()
        trial_ids = torch.zeros(batch_size, T).to(self.device).long()

        context = {
            "observations": observations_context,
            "rewards": rewards_context,
            "actions": actions_context,
            "mask": mask,
            "timesteps": timesteps,
            "trial_ids": trial_ids,
            "infos": [],
        }
        return context

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        update_time = time.time()
        total_loss = torch.zeros(self.cfg.num_train_envs, 1).to(self.device)
        # best_r = torch.zeros(self.cfg.num_train_envs, 1).to(self.device)
        best_r = torch.tensor(-float("inf")).to(self.device)

        # this is a single rollout of the policy
        # rollout N trials
        with torch.amp.autocast("cuda"):
            policy_context = self._init_context()
            for trial_id in range(self.cfg.num_trials):
                if trial_id == 0:
                    apply_meta_reset = True
                else:
                    apply_meta_reset = False

                r_explore, l_explore, policy_context = self.rollout_trial(
                    env=self.train_envs,
                    policy_type="explore",
                    context=policy_context,
                    apply_meta_reset=apply_meta_reset,
                    trial_id=trial_id,
                )
                import ipdb

                ipdb.set_trace()
                r_exploit, l_exploit, _ = self.rollout_trial(
                    env=self.train_envs,
                    policy_type="exploit",
                    context=policy_context,
                    apply_meta_reset=apply_meta_reset,
                    trial_id=trial_id,
                )

                # exploit trial is 'informative'
                # good exploit trials meet or surpass previous exploit returns in the
                # meta-rollout sequence
                mask = r_exploit >= best_r
                total_loss += l_exploit * mask

                # explore trial is 'maximal'
                # good explore trials are followed by the exploit policy achieving
                # higher trial returns than those seen so far
                mask2 = r_exploit > best_r
                total_loss += l_explore * mask2
                # best_r = r_exploit * mask + best_r * (1 - mask2.int())

                # select the best reward from all the exploit trials across environments
                r_exploit_ = r_exploit[mask2]
                # handle max of empty tensor
                if r_exploit_.numel() > 0:
                    best_r = r_exploit_.max()
                else:
                    best_r = best_r

                # log(f"Epoch: {self.current_epoch}, Best R: {best_r}")

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

    def _generate_plots(self, policy_context):
        if self.cfg.env.env_name == "bandit":
            # make a plot of rewards over time
            rewards = policy_context["rewards"]
            rewards = rewards.cpu().numpy()
            ep_ret = np.cumsum(rewards[0, :, 0], axis=0)
            plt.figure(figsize=(10, 5))
            plt.plot(ep_ret)
            # add vertical line at each episode end
            for i in range(self.cfg.num_trials):
                plt.axvline(
                    x=i * self.cfg.env.timesteps_per_trial, color="k", linestyle="--"
                )

            plt.title("Episode Return")
            plt.xlabel("Environment Steps")
            plt.ylabel("Cumulative Reward")
            plt.tight_layout()

            self.log_to_wandb({"ep_ret": wandb.Image(plt)}, prefix="plots/")
            plt.close()
        elif self.cfg.env.env_name == "darkroom":
            import ipdb

            ipdb.set_trace()

    def eval(self):
        log(
            " ======================= Running evaluation episodes ======================= ",
            color="blue",
        )
        self.model.eval()

        num_explore = 1
        num_exploit = self.cfg.num_trials - num_explore

        policy_context = self._init_context(batch_size=self.cfg.num_eval_envs)
        with torch.no_grad():
            trial_id = 0
            # we combine the exploit and explore policies for evaluation
            for _ in range(num_explore):
                r_explore, _, policy_context = self.rollout_trial(
                    env=self.eval_envs,
                    policy_type="explore",
                    context=policy_context,
                    apply_meta_reset=True,
                    trial_id=trial_id,
                )
                trial_id += 1
            for _ in range(num_exploit):
                r_exploit, _, policy_context = self.rollout_trial(
                    env=self.eval_envs,
                    policy_type="exploit",
                    context=policy_context,
                    apply_meta_reset=True,
                    trial_id=trial_id,
                )
                trial_id += 1

            ep_return = r_explore + r_exploit

            # compute mean and std of episode returns over environments
            mean_ep_return = ep_return.mean().item()
            std_ep_return = ep_return.std().item()

            # generate some visualizations of the return over time
            self._generate_plots(policy_context)

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
            # update behavior policy to be same as successor policy every T epochs
            if self.current_epoch % self.cfg.update_behavior_every == 0:
                log(
                    "Updating behavior policy to match successor policy", color="yellow"
                )
                self.model.update_behavior_policy()

            train_metrics = self.train_step()

            if self.current_epoch % self.cfg.eval_every == 0:
                eval_metrics = self.eval()

                if self.cfg.use_wandb:
                    wandb.log(eval_metrics)

                log(
                    f"Epoch {self.current_epoch}: Train Loss = {train_metrics['loss']:.4f}, Eval Mean Ep Ret = {eval_metrics['eval/mean_ep_ret']:.4f}, Eval Std Ep Ret = {eval_metrics['eval/std_ep_ret']:.4f}",
                    color="blue",
                )

        if self.wandb_run is not None:
            self.wandb_run.finish()
