import collections
from typing import Dict, List, Tuple

import numpy as np
import torch
import tqdm
from omegaconf import DictConfig

from chicken_and_egg.models.dream_dqn import DQNAgent
from chicken_and_egg.trainers.base_trainer import BaseTrainer
from chicken_and_egg.utils.data_utils import Experience, TrajectoryExperience
from chicken_and_egg.utils.logger import log


class DREAMTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.current_epoch = 0

        # Initialize metrics tracking
        self.rewards = collections.deque(maxlen=200)
        self.relabel_rewards = collections.deque(maxlen=200)
        self.exploration_lengths = collections.deque(maxlen=200)
        self.exploration_steps = 0
        self.exploitation_steps = 0

        # Setup exploration and exploitation policies after base initialization
        self.exploit_policy = self._setup_exploit_policy()
        self.exploration_policy = self._setup_exploration_policy()

    def setup_model(self):
        """Not used - policies are setup separately"""
        return None

    def _setup_exploit_policy(self) -> DQNAgent:
        """Setup the exploitation policy"""
        return DQNAgent(self.cfg.exploit_policy, self.train_envs).to(self.device)

    def _setup_exploration_policy(self) -> DQNAgent:
        """Setup the exploration policy"""
        if self.cfg.exploration_policy.type == "learned":
            return DQNAgent(self.cfg.exploration_policy, self.train_envs).to(
                self.device
            )
        else:
            raise ValueError(
                f"Invalid exploration policy type: {self.cfg.exploration_policy.type}"
            )

    @property
    def save_dict(self) -> Dict:
        """Override save_dict to include both policies"""
        return {
            "exploit_policy": {
                "model": self.exploit_policy.q_network.state_dict(),
                "target": self.exploit_policy.target_network.state_dict(),
                "optimizer": self.exploit_policy.optimizer.state_dict(),
                "updates": self.exploit_policy.updates,
                "epsilon": self.exploit_policy.epsilon,
            },
            "exploration_policy": {
                "model": self.exploration_policy.q_network.state_dict(),
                "target": self.exploration_policy.target_network.state_dict(),
                "optimizer": self.exploration_policy.optimizer.state_dict(),
                "updates": self.exploration_policy.updates,
                "epsilon": self.exploration_policy.epsilon,
            },
            "trainer_state": {
                "epoch": self.current_epoch,
                "best_reward": max(self.rewards) if self.rewards else float("-inf"),
                "exploration_steps": self.exploration_steps,
                "exploitation_steps": self.exploitation_steps,
                "rng_state": torch.get_rng_state(),
                "np_rng_state": np.random.get_state(),
            },
        }

    def load_checkpoint(self):
        """Load checkpoint using BaseTrainer's functionality"""
        checkpoint = torch.load(self.cfg.load_from_ckpt)

        # Load exploit policy
        exploit_state = checkpoint["exploit_policy"]
        self.exploit_policy.q_network.load_state_dict(exploit_state["model"])
        self.exploit_policy.target_network.load_state_dict(exploit_state["target"])
        self.exploit_policy.optimizer.load_state_dict(exploit_state["optimizer"])
        self.exploit_policy.updates = exploit_state["updates"]
        self.exploit_policy.epsilon = exploit_state["epsilon"]

        # Load exploration policy
        explore_state = checkpoint["exploration_policy"]
        self.exploration_policy.q_network.load_state_dict(explore_state["model"])
        self.exploration_policy.target_network.load_state_dict(explore_state["target"])
        self.exploration_policy.optimizer.load_state_dict(explore_state["optimizer"])
        self.exploration_policy.updates = explore_state["updates"]
        self.exploration_policy.epsilon = explore_state["epsilon"]

        # Load trainer state
        trainer_state = checkpoint["trainer_state"]
        self.current_epoch = trainer_state["epoch"]
        self.exploration_steps = trainer_state["exploration_steps"]
        self.exploitation_steps = trainer_state["exploitation_steps"]

        # Restore RNG states
        torch.set_rng_state(trainer_state["rng_state"])
        np.random.set_state(trainer_state["np_rng_state"])

        log(f"Loaded checkpoint from epoch {self.current_epoch}", color="green")

    def run_episode(self, env, policy, test=False) -> Tuple[List, List]:
        """Runs a single episode following the given policy."""
        episode = []
        renders = []
        state = env.reset()
        hidden_state = None

        while True:
            action, next_hidden_state = policy.act(state, hidden_state, test=test)
            next_state, reward, done, info = env.step(action)

            experience = Experience(
                state,
                action,
                reward,
                next_state,
                done,
                info,
                hidden_state,
                next_hidden_state,
            )
            episode.append(experience)

            state = next_state
            hidden_state = next_hidden_state

            if done:
                break

        return episode, renders

    def train_single_trial(self):
        """Run a single trial in the DREAM algorithm"""
        # Run exploration episode
        exploration_episode, _ = self.run_episode(
            self.train_envs, self.exploration_policy
        )

        # Update exploration agent
        # The reward for exploration is ||f(mu) - g(\tau_{:t})||_2 - ||f(mu) - g(\tau_{:t+1})||_2 - c
        for index, exp in enumerate(exploration_episode):
            self.exploration_policy.update(
                TrajectoryExperience(exp, exploration_episode, index)
            )

        self.exploration_steps += len(exploration_episode)
        self.exploration_lengths.append(len(exploration_episode))

        # Run exploitation episode
        # We either use the trajectory embedding or
        exploitation_episode, _ = self.run_episode(self.train_envs, self.exploit_policy)

        self.exploitation_steps += len(exploitation_episode)
        self.trajectory_embedder.use_ids(True)

        # Track metrics
        self.rewards.append(sum(exp.reward for exp in exploitation_episode))

        # Get exploration rewards
        exploration_rewards, distances = self.trajectory_embedder.label_rewards(
            [exploration_episode]
        )
        exploration_rewards = exploration_rewards[0]
        self.relabel_rewards.append(exploration_rewards.sum().item())

        # Log metrics
        metrics = {
            "reward/train": np.mean(self.rewards),
            "reward/exploration": np.mean(self.relabel_rewards),
            "steps/exploration_per_episode": np.mean(self.exploration_lengths),
            "steps/exploration": self.exploration_steps,
            "steps/exploitation": self.exploitation_steps,
        }

        # Add agent-specific metrics
        for agent_name, agent in [
            ("instruction", self.exploit_policy),
            ("exploration", self.exploration_policy),
        ]:
            for k, v in agent.stats.items():
                if v is not None:
                    metrics[f"{agent_name}_{k}"] = v

        self.log_to_wandb(metrics)
        return metrics

    def eval(self):
        """Run evaluation episodes"""
        log("Running evaluation episodes", color="blue")

        test_rewards = []
        test_exploration_lengths = []

        # We don't have task IDs during evaluation
        self.trajectory_embedder.use_ids(False)

        for test_idx in range(self.cfg.num_eval_episodes):
            # Run exploration episode
            exploration_episode, exploration_renders = self.run_episode(
                self.eval_envs, self.exploration_policy, test=True
            )
            test_exploration_lengths.append(len(exploration_episode))

            # Run exploitation episode
            episode, renders = self.run_episode(
                self.eval_envs, self.exploit_policy, test=True
            )
            test_rewards.append(sum(exp.reward for exp in episode))

        eval_metrics = {
            "eval/mean_reward": np.mean(test_rewards),
            "eval/std_reward": np.std(test_rewards),
            "eval/mean_exploration_length": np.mean(test_exploration_lengths),
        }

        self.log_to_wandb(eval_metrics)
        return eval_metrics

    def train(self):
        """Main training loop following Algorithm 2 from the paper"""
        # Load checkpoint if specified
        if self.cfg.load_from_ckpt:
            self.load_checkpoint()
        elif not self.cfg.skip_first_eval:
            self.eval()

        # Iterating over trials
        for self.current_epoch in tqdm.tqdm(
            range(self.current_epoch, self.cfg.num_epochs), desc="Training"
        ):
            train_metrics = self.train_single_trial()

            if self.current_epoch % self.cfg.eval_every == 0:
                eval_metrics = self.eval()

                # Save checkpoint using BaseTrainer's functionality
                self.save_model(self.save_dict, eval_metrics, self.current_epoch)

                log(
                    f"Epoch {self.current_epoch}: "
                    f"Train Reward = {train_metrics['reward/train']:.4f}, "
                    f"Eval Reward = {eval_metrics['eval/mean_reward']:.4f}",
                    color="blue",
                )

        if self.wandb_run is not None:
            self.wandb_run.finish()
