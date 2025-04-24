# Adapted from https://github.com/ezliu/hrl
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from chicken_and_egg.utils.data_utils import Transition


class DREAM(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        DREAM contains an exploitation policy and an exploration policy.
        Both policies are DQN agents.
        """
        super().__init__()
        self.cfg = cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exploit_policy = DQNAgent(self.cfg, device)
        self.exploration_policy = DQNAgent(self.cfg, device)


class TrajectoryEmbedder(nn.Module):
    """Trajectory embedder, embeds a sequence of transitions"""

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.state_embedder = nn.Linear(cfg.obs_dim, cfg.hidden_dim)
        self.action_embedder = nn.Linear(1, cfg.hidden_dim)
        self.reward_embedder = nn.Linear(1, cfg.hidden_dim)

        self.transition_embedder = nn.Linear(3 * cfg.hidden_dim, cfg.hidden_dim)

        self.lstm = nn.LSTM(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)

        self.output_layer = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ):
        state_embed = self.state_embedder(states)
        action_embed = self.action_embedder(actions)
        reward_embed = self.reward_embedder(rewards)

        embed = self.transition_embedder(
            torch.cat([state_embed, action_embed, reward_embed], dim=-1)
        )

        embed, hidden_state = self.lstm(embed, hidden_state)
        embed = self.output_layer(embed)

        return embed, hidden_state


class DQNAgent(nn.Module):
    """DQN Agent implementation"""

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.trajectory_embedder = TrajectoryEmbedder(cfg.trajectory_embedder)
        self.id_embedder = nn.Embedding(cfg.id_dim, cfg.hidden_dim)

        # Create networks
        self.q = DuelingQNetwork(
            num_actions=cfg.act_dim,
            input_dim=cfg.obs_dim,
            hidden_dim=cfg.hidden_dim,
            trajectory_embedder=self.trajectory_embedder,
        )

        self.target_q = DuelingQNetwork(
            num_actions=cfg.act_dim,
            input_dim=cfg.obs_dim,
            hidden_dim=cfg.hidden_dim,
            trajectory_embedder=self.trajectory_embedder,
        )

        # Sync target network initially
        self.sync_target()

    def label_rewards(
        self, traj_batched: List[Transition], env_ids: torch.Tensor, mask: torch.Tensor
    ):
        """Computes rewards for each experience in the trajectory"""

        traj = [traj[0] for traj in traj_batched]
        # [B, T, N, O]
        obs = torch.tensor([t.obs for t in traj], device=self.device).float()
        # [B, T, N]
        actions = torch.tensor([t.action for t in traj], device=self.device).float()
        # [B, T, N]
        rewards = torch.tensor([t.reward for t in traj], device=self.device).float()

        # [B, T, D]
        transition_contexts, _ = self.trajectory_embedder(
            obs.squeeze(2), actions, rewards
        )
        import ipdb

        ipdb.set_trace()
        id_contexts = self.id_embedder(env_ids.long())

        distances = (
            (
                transition_contexts
                - id_contexts.unsqueeze(1).expand_as(transition_contexts).detach()
            )
            ** 2
        ).sum(-1)
        # Add penalty
        rewards = distances[:, :-1] - distances[:, 1:] - self.cfg.penalty
        return (rewards * mask[:, 1:]).detach(), distances

    def _compute_loss(
        self, batch: List[Transition], mask: torch.Tensor, relabel_rewards: bool = False
    ):
        """Compute standard Double DQN loss

        dqn loss = (r + gamma * max_a Q(s', a') - Q(s, a))**2

        Args:
            batch: List[Transition]
            mask: torch.Tensor
        """
        with torch.no_grad():
            # [B, T, N, A]
            next_q_values, _ = self.target_q(batch.next_obs)
            # [B, T, N]
            next_action = next_q_values.argmax(dim=-1)

        # [B, T, N, A]
        q_values, _ = self.q(batch.obs)
        # [B, T, N, A]
        q_values_next, _ = self.q(batch.next_obs)

        # [B, T, N, 1]
        q_values_next_target = q_values_next.gather(
            -1, next_action.unsqueeze(-1)
        ).squeeze(-1)

        rewards = batch.reward

        if relabel_rewards:
            rewards = self.label_rewards(
                batch.trajectory, env_ids=batch.env_id, mask=mask
            )

        # [B, T, N]
        target_q_values = rewards + self.cfg.gamma * q_values_next_target

        current_q_values = q_values.gather(
            -1, batch.action.long().unsqueeze(-1)
        ).squeeze(-1)

        # [B, T, N]
        loss = F.mse_loss(current_q_values, target_q_values, reduction="none")

        # [B, T, N]
        weights = mask.float().unsqueeze(-1)  # for the env dimension
        loss = loss * weights
        loss = loss.sum() / mask.sum()

        return loss

    def select_action(
        self,
        obs: np.ndarray,
        hidden_state: Optional[torch.Tensor] = None,
        test: bool = False,
    ) -> Tuple[int, None]:
        """Select action using epsilon-greedy policy

        Args:
            obs: [B, obs_dim]
        """
        obs_tensor = torch.from_numpy(obs).to(self.device).float()
        q_values, hidden_state = self.q(obs_tensor, hidden_state)

        # Use smaller epsilon during testing
        epsilon = self.cfg.test_epsilon if test else self.cfg.epsilon

        # Epsilon-greedy action selection
        if np.random.random() > epsilon:
            action = q_values.argmax(dim=1)
        else:
            action = np.random.randint(0, self.cfg.act_dim, size=obs.shape[0])

        return action, None

    def sync_target(self):
        """Sync target network with current Q network"""
        self.target_q.load_state_dict(self.q.state_dict())


class DuelingQNetwork(nn.Module):
    """Dueling DQN Network Architecture with MLP obs embedder"""

    def __init__(self, num_actions: int, input_dim: int, hidden_dim: int = 64):
        super().__init__()

        # Simple MLP obs embedder
        self.obs_embedder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )

        self.obs_embedder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Advantage and value heads
        self.advantage = nn.Linear(hidden_dim, num_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass

        Args:
            obs: Batch of obs (B, T, N, D)
            hidden_state: Not used, kept for compatibility

        Returns:
            q_values: Q-values for each action
            next_hidden: Always None for MLP
        """
        # Get obs embeddings
        obs_embed = self.obs_embedder(obs)
        obs_embed, hidden_state = self.obs_embedder_lstm(obs_embed, hidden_state)

        # Get trajectory embed! I think this is for the exploit policy
        # but not sure

        # Compute advantage and value streams
        advantage = self.advantage(obs_embed)
        value = self.value(obs_embed)

        # Combine using dueling formula
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values, None
