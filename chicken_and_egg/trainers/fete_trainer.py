import time
from pathlib import Path
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
        trial_id: int = None,
        sample_actions: bool = True,
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

        obs, info = env.reset(options={"reset_task": False})
        obs = torch.from_numpy(obs).to(self.device)

        # add the new observations to the context
        observations_context = torch.roll(observations_context, shifts=-1, dims=1)
        observations_context[:, -1] = obs
        # roll mask
        mask = torch.roll(mask, shifts=-1, dims=1)
        mask[:, -1] = 1
        # set the last mask to 1

        # roll trial_id
        trial_ids = torch.roll(trial_ids, shifts=-1, dims=1)
        trial_ids[:, -1] = trial_id

        timesteps = torch.roll(timesteps, shifts=-1, dims=1)
        timesteps[:, -1] = 0

        # roll rewards and actions
        rewards_context = torch.roll(rewards_context, shifts=-1, dims=1)
        actions_context = torch.roll(actions_context, shifts=-1, dims=1)
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

            # compute loss for current timestep
            logits_t = logits[:, -1]
            # apply softmax to get a valid distribution
            logits_softmaxed = F.softmax(logits_t, dim=-1)

            if sample_actions:
                # treat as weights
                action_t = torch.multinomial(logits_softmaxed, num_samples=1).squeeze()
            else:
                action_t = torch.argmax(logits_t, dim=-1)

            # cross entropy loss
            action_loss += F.cross_entropy(logits_t, action_t, reduction="mean")

            next_state, reward, done, terminal, info = env.step(to_numpy(action_t))

            next_state = torch.from_numpy(next_state).to(self.device)
            action_t = action_t.float().detach()
            action_t = einops.repeat(action_t, "b -> b t", t=1)

            reward = torch.from_numpy(reward).to(self.device).unsqueeze(-1).float()

            # update context by appending new state, reward and action

            # if its the last timestep, don't append the next state
            if ts < self.cfg.env.timesteps_per_trial - 1:
                observations_context = torch.roll(
                    observations_context, shifts=-1, dims=1
                )
                observations_context[:, -1] = next_state
                mask = torch.roll(mask, shifts=-1, dims=1)
                mask[:, -1] = 1

                # add the trial id to the context
                trial_id_tensor = torch.tensor(trial_id).long()
                trial_id_tensor = einops.repeat(
                    trial_id_tensor, " -> b", b=env.num_envs
                ).to(self.device)
                trial_ids = torch.roll(trial_ids, shifts=-1, dims=1)
                trial_ids[:, -1] = trial_id_tensor
                timesteps = torch.roll(timesteps, shifts=-1, dims=1)
                timesteps[:, -1] = ts + 1
                infos.append(info)
                # apply roll here is to account for the
                # first observation which doesn't have an associated reward / action
                rewards_context[:, -1] = reward
                rewards_context = torch.roll(rewards_context, shifts=-1, dims=1)
                actions_context[:, -1] = action_t
                actions_context = torch.roll(actions_context, shifts=-1, dims=1)
            else:
                # just insert action and reward
                rewards_context[:, -1] = reward
                actions_context[:, -1] = action_t

            trial_return += reward

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

        # average loss across the environments
        action_loss /= env.num_envs
        return trial_return, action_loss, new_context

    def _init_context(self, batch_size: int = 1):
        T = self.cfg.env.timesteps_per_trial * self.cfg.num_trials
        # add some dummy timesteps for the initial obs
        # T += self.cfg.num_trials
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
        """
        Runs a single training step. This is a single rollout (episode) of the policy which consists of
        1 trials followed by 1 exploit trial.

        Only the explore trial is added as context.
        """
        self.model.train()
        self.optimizer.zero_grad()

        update_time = time.time()
        total_loss = torch.zeros(self.cfg.num_train_envs, 1).to(self.device)
        explore_loss = torch.zeros(self.cfg.num_train_envs, 1).to(self.device)
        exploit_loss = torch.zeros(self.cfg.num_train_envs, 1).to(self.device)

        # best_r = torch.zeros(self.cfg.num_train_envs, 1).to(self.device)

        # reset best_r every epoch
        self.best_r = torch.tensor(0.0).to(self.device)

        # this is a single rollout of the policy
        # rollout N trials
        with torch.amp.autocast("cuda"):
            policy_context = self._init_context(self.cfg.num_train_envs)
            for trial_id in range(self.cfg.num_trials):
                # run one trial of explore policy
                # and add this to the context for the exploit policy
                r_explore, l_explore, policy_context = self.rollout_trial(
                    env=self.train_envs,
                    policy_type="explore",
                    context=policy_context,
                    trial_id=trial_id,
                    sample_actions=True,
                )
                # run one trial of exploit policy which is used as feedback
                # to train the explore policy
                r_exploit, l_exploit, _ = self.rollout_trial(
                    env=self.train_envs,
                    policy_type="exploit",
                    context=policy_context,
                    trial_id=trial_id + 1,
                    sample_actions=True,
                )

                # exploit trial is 'informative'
                # good exploit trials meet or surpass previous exploit returns in the
                # meta-rollout sequence
                # train the explot policy here
                mask = r_exploit >= self.best_r
                total_loss += l_exploit * mask
                exploit_loss += l_exploit * mask

                # explore trial is 'maximal'
                # good explore trials are followed by the exploit policy achieving
                # higher trial returns than those seen so far
                # train the explore policy here
                mask2 = r_exploit > self.best_r
                total_loss += l_explore * mask2
                explore_loss += l_explore * mask2
                # best_r = r_exploit * mask + best_r * (1 - mask2.int())

                # select the best reward from all the exploit trials across environments
                r_exploit_ = r_exploit[mask2]
                # handle max of empty tensor

                # update the baseline return
                if r_exploit_.numel() > 0:
                    self.best_r = r_exploit_.max()
                else:
                    self.best_r = self.best_r

                # log(f"Epoch: {self.current_epoch}, Best R: {best_r}")

        # average loss over number of environments
        total_loss = total_loss.mean()
        explore_loss = explore_loss.mean()
        exploit_loss = exploit_loss.mean()
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
            "explore_loss": explore_loss.item(),
            "exploit_loss": exploit_loss.item(),
            "best_r": self.best_r.item(),
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
            self._visualize_return(policy_context)
            self._visualize_darkroom_traj(policy_context)

    def _visualize_return(self, policy_context):
        rewards = policy_context["rewards"]  # [N, T, 1]
        trial_ids = policy_context["trial_ids"]  # [N, T]

        # For each environment
        num_save = min(self.cfg.num_eval_rollouts_save, rewards.shape[0])
        for env_idx in range(num_save):
            # Get rewards for this environment
            rewards_env = rewards[env_idx].cpu().numpy()
            trial_ids_env = trial_ids[env_idx].cpu().numpy()

            # Calculate cumulative return
            ep_ret = np.cumsum(rewards_env, axis=0)

            # Create figure
            plt.figure(figsize=(12, 6))

            # Adjust subplot margins to make room for title
            plt.subplots_adjust(top=0.85)

            # Plot cumulative return
            plt.plot(ep_ret, "b-", label="Cumulative Return")

            # Add vertical line after explore is over
            plt.axvline(
                x=self.cfg.num_eval_explore_trials * self.cfg.env.timesteps_per_trial,
                color="k",
                linestyle="--",
            )

            # Add vertical lines at trial boundaries
            unique_trials = np.unique(trial_ids_env)
            for trial_id in unique_trials[1:]:  # Skip first boundary
                # Find first occurrence of this trial_id
                trial_boundary = np.where(trial_ids_env == trial_id)[0][0]
                plt.axvline(x=trial_boundary, color="r", linestyle="--", alpha=0.5)
                # Add trial number
                plt.text(
                    trial_boundary,
                    plt.ylim()[1],
                    f"T{trial_id}",
                    rotation=0,
                    ha="right",
                    va="bottom",
                )

            # Use suptitle instead of title to place it higher
            plt.suptitle(f"Cumulative Reward - Env {env_idx}", y=0.95)
            plt.xlabel("Steps")
            plt.ylabel("Cumulative Return")
            plt.grid(True, alpha=0.3)

            # Save plot
            plot_path = Path(self.cfg.exp_dir) / "ep_ret" / f"ep_ret_{env_idx}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, bbox_inches="tight")
            log(f"Saved return plot to {plot_path}", color="green")

            # Log to wandb
            self.log_to_wandb({f"ep_ret_{env_idx}": wandb.Image(plt)}, prefix="ep_ret/")
            plt.close()

    def _visualize_darkroom_traj(self, policy_context):
        """Generate grid-based visualization of darkroom environment showing agent trajectory."""
        try:
            from io import BytesIO

            import imageio
        except ImportError:
            log(
                "Please install imageio: pip install imageio imageio-ffmpeg",
                color="red",
            )
            return

        # Extract relevant information
        observations = policy_context["observations"]  # [N, T, 2]
        rewards = policy_context["rewards"]  # [N, T, 1]
        trial_ids = policy_context["trial_ids"]  # [N, T]
        actions = policy_context["actions"]  # [N, T, 1]
        infos = policy_context["infos"]  # List of dicts

        ACTION_MAP = {0: "No-op", 1: "Up", 2: "Right", 3: "Down", 4: "Left"}

        num_save = min(self.cfg.num_eval_rollouts_save, rewards.shape[0])

        for env_idx in range(num_save):
            video_start = time.time()
            # Get first info that contains reward information
            env_info = None
            for info in infos:
                if (
                    isinstance(info, dict)
                    and "rx" in info
                    and "ry" in info
                    and "rr" in info
                ):
                    env_info = info
                    break

            if env_info is None:
                continue

            # Setup grid dimensions
            w, h = int(env_info["w"][env_idx]), int(env_info["h"][env_idx])

            rr = env_info["rr"][env_idx]

            # Get agent trajectory
            obs = observations[env_idx].cpu().numpy()  # obs comes as [x, y]
            rewards_env = rewards[env_idx].cpu().numpy()
            ret = np.cumsum(rewards_env, axis=0)

            # Sample frames
            max_frames = 100
            step = max(1, len(obs) // max_frames)
            frame_indices = list(range(0, len(obs), step))
            if len(obs) - 1 not in frame_indices:
                frame_indices.append(len(obs) - 1)

            frames = []

            # Generate frames
            for t in frame_indices:
                # Create visualization grid for this frame
                grid = np.zeros((h, w))  # Use (h, w) for matrix indexing
                visited_grid = np.zeros((h, w))
                visited = infos[t]["visited"][env_idx]

                # Add rewards to grid, distinguishing between visited and unvisited
                for i, (x, y, r) in enumerate(
                    zip(
                        env_info["rx"][env_idx],
                        env_info["ry"][env_idx],
                        env_info["rr"][env_idx],
                    )
                ):
                    if visited[i]:
                        visited_grid[y, x] = r  # Use [y, x] for matrix indexing
                    else:
                        grid[y, x] = r  # Use [y, x] for matrix indexing

                # Add agent's past positions (value = -0.5)
                for past_t in range(t):
                    x, y = obs[past_t].astype(int)  # obs comes as [x, y]
                    if 0 <= x < w and 0 <= y < h:
                        if grid[y, x] == 0 and visited_grid[y, x] == 0:  # Use [y, x]
                            grid[y, x] = -0.5  # Use [y, x]

                # Add current agent position (value = -1)
                x, y = obs[t].astype(int)
                if 0 <= x < w and 0 <= y < h:
                    grid[y, x] = -1  # Use [y, x]

                # Add start position if not already marked (value = -0.75)
                start_x, start_y = obs[0].astype(int)
                if grid[start_y, start_x] == -0.5:  # Use [y, x]
                    grid[start_y, start_x] = -0.75  # Use [y, x]

                fig = plt.figure(figsize=(8, 8))

                # Create custom colormap for unvisited treasures and agent
                colors = ["blue", "green", "lightblue", "white", "yellow", "red"]
                nodes = [-1, -0.75, -0.5, 0, 0.5, 1]
                cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
                    "custom", list(zip(np.linspace(0, 1, len(nodes)), colors))
                )

                # Plot the base grid
                plt.imshow(
                    grid,
                    cmap=cmap,
                    interpolation="nearest",
                    vmin=-1,
                    vmax=1,
                    origin="upper",
                )

                # Overlay visited treasures
                visited_mask = visited_grid != 0
                if visited_mask.any():
                    plt.imshow(
                        np.ma.masked_where(~visited_mask, visited_grid),
                        cmap=plt.cm.Greys,
                        interpolation="nearest",
                        alpha=0.7,
                        vmin=-1,
                        vmax=1,
                        origin="upper",
                    )

                # Add reward values as text for all treasures (both visited and unvisited)
                for i, (rx, ry, rr) in enumerate(
                    zip(
                        env_info["rx"][env_idx],
                        env_info["ry"][env_idx],
                        env_info["rr"][env_idx],
                    )
                ):
                    plt.text(
                        rx,
                        ry - 0.2,  # Slightly above the cell
                        f"{rr:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="black",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                    )

                # Add text
                current_ret = float(ret[t, 0]) if t < len(ret) else float(ret[-1, 0])
                current_trial = int(trial_ids[env_idx, t].cpu().item())
                action = int(actions[env_idx, t].cpu().item())
                plt.title(
                    f"Step: {t}, Return: {current_ret:.2f}, Trial: {current_trial}\n"
                    f"Action: {ACTION_MAP[action]} ({action}), Pos: ({x}, {y})"
                )

                # Save figure to buffer
                buf = BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)

                # Read image from buffer
                frame = imageio.imread(buf)
                frames.append(frame)

                # Cleanup
                plt.close()
                buf.close()

            video_time = time.time() - video_start
            log(f"Video generation time: {video_time:.2f} seconds", color="green")

            # Save video
            video_path = (
                Path(self.cfg.exp_dir) / "eval_rollouts" / f"rollout_{env_idx}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                imageio.mimsave(str(video_path), frames, fps=1)
                log(f"Saved rollout animation to {video_path}", color="green")

                # Log to wandb
                self.log_to_wandb(
                    {f"rollout_{env_idx}": wandb.Video(str(video_path))},
                    prefix="plots/",
                )
            except Exception as e:
                log(f"Failed to save video: {str(e)}", color="red")

    def eval(self):
        log(
            " ======================= Running evaluation episodes ======================= ",
            color="blue",
        )
        self.model.eval()

        num_explore = self.cfg.num_eval_explore_trials
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
                    trial_id=trial_id,
                    sample_actions=False,
                )
                trial_id += 1

            for _ in range(num_exploit):
                r_exploit, _, policy_context = self.rollout_trial(
                    env=self.eval_envs,
                    policy_type="exploit",
                    context=policy_context,
                    trial_id=trial_id,
                    sample_actions=False,
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
