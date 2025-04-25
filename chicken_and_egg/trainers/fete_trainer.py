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
from chicken_and_egg.utils.general_utils import compute_entropy, to_numpy
from chicken_and_egg.utils.logger import log


class FETETrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.current_epoch = 0

    def setup_model(self):
        model = FETE(self.cfg.model)
        return model

    def rollout_episode(
        self,
        env: gym.Env,
        policy_type: str,
        context: Dict[str, torch.Tensor] = None,
        trial_id: int = None,
        sample_actions: bool = True,
    ):
        """
        Rollout a trial for the GPT-2 based transformer policy.

        Returns:
            ep_ret: float
            action_loss: float
            context: Dict[str, torch.Tensor], [N, T, ...]
        """        
        # for exploitation, use the context from the exploration policy
        observations_context = context["observations"]
        rewards_context = context["rewards"]
        actions_context = context["actions"]
        mask = context["mask"]
        timesteps = context["timesteps"]
        episode_ids = context["episode_ids"]
        infos = context["infos"].copy()  # Create a copy to avoid modifying the original

        if trial_id == 0:
            reset_task = True
        else:
            reset_task = False

        obs, info = env.reset(options={"reset_task": reset_task})
        obs = torch.from_numpy(obs).to(self.device)

        # add the new observations to the context
        observations_context = torch.roll(observations_context, shifts=-1, dims=1)
        observations_context[:, -1] = obs
        
        # roll mask
        mask = torch.roll(mask, shifts=-1, dims=1)
        mask[:, -1] = 1
        # set the last mask to 1

        # roll episode ids
        episode_ids = torch.roll(episode_ids, shifts=-1, dims=1)
        episode_ids[:, -1] = trial_id

        timesteps = torch.roll(timesteps, shifts=-1, dims=1)
        timesteps[:, -1] = 0

        # roll rewards and actions
        rewards_context = torch.roll(rewards_context, shifts=-1, dims=1)
        actions_context = torch.roll(actions_context, shifts=-1, dims=1)
        infos.append(info)

        # Initialize lists to store entropy values across timesteps
        behavior_entropies = []
        successor_entropies = []

        # Accumulate per environment 
        action_loss = torch.zeros(env.num_envs,).to(self.device)
        ep_ret = torch.zeros(env.num_envs, 1).to(self.device)

        for ts in range(self.cfg.env.timesteps_per_episode):
            # [N, T, A], do not update the gradients for the behavior policy
            # these will be updated by copying the weights from the successor policy
            policy_kwargs = {
                "observations": observations_context,
                "actions": actions_context,
                "rewards": rewards_context,
                "timesteps": timesteps,
                # "attention_mask": ~mask.bool(),  # be careful here 1 means we mask and 0 means we attend
                "attention_mask": mask,
                "episode_ids": episode_ids,
            }

            # don't update the behavior policy, see paper
            with torch.no_grad():
                behavior_logits = self.model(
                    **policy_kwargs, policy_type=f"{policy_type}_roll"
                )

            # [N, T, A]
            successor_logits = self.model(
                **policy_kwargs, policy_type=f"{policy_type}_pred"
            )

            # compute loss for current timestep
            # this is for taking a step in the environment
            logits_t = behavior_logits[:, -1]
            
            # Apply temperature scaling
            if sample_actions:
                temperature = self.cfg.eval_temperature if not self.model.training else self.cfg.temperature
                # if "exploit" in policy_type:
                #     temperature = temperature * 0.5  # Lower temperature for exploit

                logits_t = logits_t / temperature
            else:
                # regular softmax
                temperature = 1.0

            # compute entropy of logits
            current_successor_entropy = compute_entropy(successor_logits[:, -1] / temperature)
            successor_entropies.append(current_successor_entropy)

            # compute entropy of behavior logits
            current_behavior_entropy = compute_entropy(behavior_logits[:, -1] / temperature)
            behavior_entropies.append(current_behavior_entropy)

            # apply softmax to get a valid distribution
            logits_softmaxed = F.softmax(logits_t, dim=-1)

            if sample_actions:
                # treat as weights
                action_t = torch.multinomial(logits_softmaxed, num_samples=1).squeeze()
            else:
                # this is eval, use the successor logits
                # we use deterministic execution, but don't combine the logits between
                # behavior and successor policies
                logits_eval = behavior_logits[:, -1]
                logits_softmax = F.softmax(logits_eval, dim=-1)
                action_t = torch.argmax(logits_softmax, dim=-1)

            # Add logits instead of multiplying (since they're in log space)
            # When we train, we compare against the sum of the two logits
            logits_train = behavior_logits + successor_logits
            logits_train = logits_train[:, -1]

            # cross entropy loss
            if action_t.ndim == 0:
                action_t = action_t.unsqueeze(0)
                current_loss = F.cross_entropy(logits_train, action_t, reduction="none")
            else:
                current_loss = F.cross_entropy(logits_train, action_t, reduction="none")

            action_loss += current_loss

            next_state, reward, done, terminal, info = env.step(to_numpy(action_t))

            next_state = torch.from_numpy(next_state).to(self.device)
            action_t = action_t.float().detach()
            action_t = einops.repeat(action_t, "b -> b t", t=1)

            reward = torch.from_numpy(reward).to(self.device).unsqueeze(-1).float()

            # Store current action and reward before rolling for next timestep
            actions_context[:, -1] = action_t
            rewards_context[:, -1] = reward
            
            ep_ret += reward

            # assumes all envs finish at the same time
            if done.all():
                break

            # update context by appending new state, reward and action
            if ts < self.cfg.env.timesteps_per_episode - 1:
                observations_context = torch.roll(observations_context, shifts=-1, dims=1)
                observations_context[:, -1] = next_state
                
                mask = torch.roll(mask, shifts=-1, dims=1)
                mask[:, -1] = 1

                # add the trial id to the context
                episode_ids = torch.roll(episode_ids, shifts=-1, dims=1)
                episode_ids[:, -1] = trial_id
                
                timesteps = torch.roll(timesteps, shifts=-1, dims=1)
                timesteps[:, -1] = ts + 1
                
                # Roll actions and rewards AFTER setting their values
                actions_context = torch.roll(actions_context, shifts=-1, dims=1)
                rewards_context = torch.roll(rewards_context, shifts=-1, dims=1)
                
                infos.append(info)

        # Calculate mean entropies across all timesteps
        successor_entropy_mean = torch.cat(successor_entropies).mean().item()
        behavior_entropy_mean = torch.cat(behavior_entropies).mean().item()

        new_context = {
            "observations": observations_context,
            "rewards": rewards_context,
            "actions": actions_context,
            "mask": mask,
            "timesteps": timesteps,
            "episode_ids": episode_ids,
            "infos": infos,
        }

        episode_metrics = {
            "successor_entropy": successor_entropy_mean,
            "behavior_entropy": behavior_entropy_mean,
            "ep_ret": ep_ret.mean().item(),
        }

        num_timesteps = min(self.cfg.env.timesteps_per_episode, ts + 1)
        action_loss /= num_timesteps
        # take mean over environments
        action_loss = action_loss.mean()
        
        return ep_ret, action_loss, new_context, episode_metrics

    def _init_context(self, batch_size: int = 1):
        T = self.cfg.env.timesteps_per_episode * self.cfg.num_episodes
        # add some dummy timesteps for the initial obs
        # T += self.cfg.num_episodes
        # T += 1  # for padding (?)

        # keep track of context here for observation, reward and action
        O = self.cfg.env.obs_dim
        A = self.cfg.env.act_dim

        # create context for the behavior explore/exploit policy
        observations_context = torch.zeros(batch_size, T, O).to(self.device)
        rewards_context = torch.zeros(batch_size, T, 1).to(self.device)
        actions_context = torch.zeros(batch_size, T, 1).to(self.device)
        mask = torch.zeros(batch_size, T).to(self.device)
        timesteps = torch.zeros(batch_size, T).to(self.device).long()
        episode_ids = torch.zeros(batch_size, T).to(self.device).long()

        context = {
            "observations": observations_context,
            "rewards": rewards_context,
            "actions": actions_context,
            "mask": mask,
            "timesteps": timesteps,
            "episode_ids": episode_ids,
            "infos": [],
        }
        return context

    def run_single_trial(self, stage: str = "train"):

        num_envs = self.cfg.num_train_envs if stage == "train" else self.cfg.num_eval_envs

        update_time = time.time()
        total_loss = torch.zeros(num_envs, 1).to(self.device)
        explore_loss = torch.zeros(num_envs, 1).to(self.device)
        exploit_loss = torch.zeros(num_envs, 1).to(self.device)

        # reset best_r for each parallel environment
        # best_r is also reset every trial
        best_r = torch.zeros(num_envs, 1).to(self.device)

        trial_metrics = []
        best_r_history = []  # Track best_r over time
        best_r_diffs = []    # Track improvements in best_r

        # this is a single rollout of the policy
        # rollout N episodes

        num_episodes = self.cfg.num_episodes - 1 if stage == "train" else self.cfg.num_episodes
        env = self.train_envs if stage == "train" else self.eval_envs

        ep_ret = torch.zeros(num_envs, 1).to(self.device)

        with torch.amp.autocast("cuda"):
            policy_context = self._init_context(num_envs)

            # If we want N episodes, we need to rollout N-1 explore / exploit pairs.
            for trial_id in range(num_episodes):
                if stage == "train" or (stage == "eval" and trial_id < 1):
                    # run one trial of explore policy
                    # and add this to the context for the exploit policy
                    r_explore, l_explore, policy_context, episode_metrics = (
                        self.rollout_episode(
                            env=env,
                            policy_type="explore",
                            context=policy_context,
                            trial_id=trial_id,
                            sample_actions=stage == "train",
                        )
                    )

                    ep_ret += r_explore

                # run one trial of exploit policy which is used as feedback
                # to train the explore policy
                # NOTE: during training, we don't include the context from the exploit policy
                # so these transitions are ignored
                # pc_tmp is the context from the explore policy but is not used during training
                r_exploit, l_exploit, pc_tmp, episode_metrics = self.rollout_episode(
                    env=env,
                    policy_type="exploit",
                    context=policy_context,
                    trial_id=trial_id + 1,
                    sample_actions=stage == "train",
                )
                ep_ret += r_exploit

                # if we are in eval, we need to use the context from the exploit policy
                # for the next trial
                if stage == "eval":
                    policy_context = pc_tmp

                # Calculate masks based on current state rewards
                mask = r_exploit >= best_r
                if self.cfg.weighting:
                    mask = mask * (1 + r_exploit - best_r)
                
                # Update best_r and track improvement
                prev_best_r = best_r.clone()
                best_r = torch.max(best_r, r_exploit)
                best_r_diff = best_r - prev_best_r
                
                # Store metrics
                best_r_history.append(best_r.mean().item())
                best_r_diffs.append(best_r_diff.mean().item())

                # Apply losses
                total_loss += l_exploit * mask
                exploit_loss += l_exploit * mask
                total_loss += l_explore * mask
                explore_loss += l_explore * mask

                # Log detailed metrics for this trial
                trial_metrics.append({
                    **episode_metrics,
                    "explore_return": r_explore.mean().item(),
                    "exploit_return": r_exploit.mean().item(),
                    "best_r": best_r.mean().item(),
                    "best_r_diff": best_r_diff.mean().item(),
                    "explore_loss": l_explore.mean().item(),
                    "exploit_loss": l_exploit.mean().item(),
                    "mask_mean": mask.float().mean().item(),
                })
    
            
        metrics = {}
        metrics["time/update"] = time.time() - update_time
        metrics["lr"] = self.scheduler.get_last_lr()[0]
        metrics[f"{stage}/mean_ep_ret"] = ep_ret.mean().item()
        metrics[f"{stage}/std_ep_ret"] = ep_ret.std().item()

        # Log per-trial metrics
        for i, trial_metric in enumerate(trial_metrics):
            for k, v in trial_metric.items():
                metrics[f"t-{stage}_{k}/trial{i}"] = v

        # Log best_r history and diffs
        for i, (best_r_val, best_r_diff) in enumerate(zip(best_r_history, best_r_diffs)):
            metrics[f"t-{stage}_best_r_history/trial{i}"] = best_r_val
            metrics[f"t-{stage}_best_r_diffs/trial{i}"] = best_r_diff
        
        # average loss over number of environments
        total_loss = total_loss.mean()
        explore_loss = explore_loss.mean()
        exploit_loss = exploit_loss.mean()

        # Log summary statistics
        metrics.update({
            f"{stage}/loss": total_loss.item(),
            f"{stage}/explore_loss": explore_loss.item(),
            f"{stage}/exploit_loss": exploit_loss.item(),
            f"{stage}/final_best_r": best_r.mean().item(),
            f"{stage}/mean_best_r_diff": np.mean(best_r_diffs),
            f"{stage}/max_best_r_diff": np.max(best_r_diffs),
            f"{stage}/num_improvements": sum(1 for diff in best_r_diffs if diff > 0),
        })

        return metrics, total_loss, policy_context

    def update(self, stage: str = "train"):
        """
        Runs a single training step. This is a single rollout (episode) of the policy which consists of
        1 trials followed by 1 exploit trial.

        Only the explore trial is added as context.
        """
        self.model.train()
        self.optimizer.zero_grad()
        metrics, total_loss, policy_context  = self.run_single_trial(stage=stage)

        self.scaler.scale(total_loss).backward()
        # Unscale gradients to prepare for gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.cfg.clip_grad_norm
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return metrics, policy_context


    def _generate_plots(self, policy_context):
        if self.cfg.env.env_name == "bandit":
            # make a plot of rewards over time
            rewards = policy_context["rewards"]
            rewards = rewards.cpu().numpy()
            ep_ret = np.cumsum(rewards[0, :, 0], axis=0)
            plt.figure(figsize=(10, 5))
            plt.plot(ep_ret)
            # add vertical line at each episode end
            for i in range(self.cfg.num_episodes):
                plt.axvline(
                    x=i * self.cfg.env.timesteps_per_episode, color="k", linestyle="--"
                )

            plt.title("Episode Return")
            plt.xlabel("Environment Steps")
            plt.ylabel("Cumulative Reward")
            plt.tight_layout()

            self.log_to_wandb({"ep_ret": wandb.Image(plt)}, prefix="plots/")
            plt.close()
        elif self.cfg.env.env_name == "darkroom":
            # pass
            # self._visualize_return(policy_context)
            self._visualize_darkroom_traj(policy_context)

    def _visualize_coverage_map(self, policy_context):
        from cae_commons.viz.darkroom import visualize_coverage_map

        videos = visualize_coverage_map(policy_context, self.cfg.num_eval_rollouts_save)
        render_videos = []
        for i, video in enumerate(videos):
            video = np.array(video)
            video = video.transpose(0, 3, 1, 2)  # HWC -> CHW
            render_videos.append(
                wandb.Video(video, caption=f"Eval Rollout {i}", fps=10, format="mp4")
            )

        if self.wandb_run is not None:
            self.wandb_run.log({"eval/coverage_maps": render_videos})
        return render_videos

    def _visualize_return(self, policy_context):
        from cae_commons.viz.darkroom import visualize_return

        images = visualize_return(
            policy_context,
            self.cfg.num_eval_rollouts_save,
            self.cfg.num_episodes,
            self.cfg.env.timesteps_per_episode,
        )

        render_images = []
        for i, image in enumerate(images):
            render_images.append(wandb.Image(image, caption=f"Eval Rollout {i}"))

        if self.wandb_run is not None:
            self.wandb_run.log({"eval/return_plots": render_images})
        return render_images

    def _visualize_darkroom_traj(self, policy_context):
        # this should be a list of videos
        from cae_commons.viz.darkroom import visualize_darkroom_traj

        videos = visualize_darkroom_traj(
            policy_context, self.cfg.num_eval_rollouts_save
        )

        render_videos = []
        for i, video in enumerate(videos):
            video = np.array(video)[:, :, :-1]  # remove alpha channel
            video = video.transpose(0, 3, 1, 2)  # HWC -> CHW
            render_videos.append(
                wandb.Video(video, caption=f"Eval Rollout {i}", fps=10, format="mp4")
            )

        if self.wandb_run is not None:
            self.wandb_run.log({"eval/trajectories": render_videos})
        return render_videos

    def eval(self):
        log(
            " ======================= Running evaluation episodes ======================= ",
            color="blue",
        )
        self.model.eval()

        with torch.no_grad():
            eval_metrics, _, policy_context  = self.run_single_trial(stage="eval")

        self.log_to_wandb(eval_metrics, prefix="")

        # generate some visualizations of the return over time
        self._generate_plots(policy_context)
        return eval_metrics

    def train(self):
        if not self.cfg.skip_first_eval:
            eval_metrics = self.eval()

        total_timesteps = 0
        start_time = time.time()
        for self.current_epoch in tqdm.tqdm(
            range(self.cfg.num_epochs), desc="Training", total=self.cfg.num_epochs
        ):
            # update behavior policy to be same as successor policy every T epochs
            if self.current_epoch % self.cfg.update_behavior_every == 0:
                log(
                    "Updating behavior policy to match successor policy", color="yellow"
                )
                self.model.update_behavior_policy()

            trial_start = time.time()
            train_metrics, policy_context = self.update(stage="train")
            trial_end = time.time()

            total_timesteps += (
                self.cfg.env.timesteps_per_episode
                * self.cfg.num_train_envs
                * self.cfg.env.num_episodes
            )

            fps = total_timesteps / (time.time() - start_time)

            train_metrics["time/trial_time"] = trial_end - trial_start
            train_metrics["time/fps"] = fps
            self.log_to_wandb(train_metrics, prefix="")
            self.log_to_wandb({"_update": self.current_epoch}, prefix="step/")

            if self.current_epoch % self.cfg.eval_every == 0:
                eval_metrics = self.eval()

                if self.cfg.use_wandb:
                    wandb.log(eval_metrics)

                log(
                    f"E {self.current_epoch}, T {total_timesteps}, FPS {fps:.2f}: Train Loss = {train_metrics['train/loss']:.4f}, Eval Mean Ep Ret = {eval_metrics['eval/mean_ep_ret']:.4f}, Eval Std Ep Ret = {eval_metrics['eval/std_ep_ret']:.4f}",
                    color="blue",
                )

        if self.wandb_run is not None:
            self.wandb_run.finish()
