# Adapted from https://github.com/ezliu/hrl
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

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
        self.exploit_policy = DQNAgent(self.cfg.kwargs, device)
        self.exploration_policy = DQNAgent(self.cfg.kwargs, device)

    def update_exploration(self, batch: List[Transition]):
        """Update exploration policy using batch of transitions"""

        import ipdb

        ipdb.set_trace()
        # convert list of transitions to Transition object
        batch = Transition(
            obs=torch.stack([t.obs for t in batch]),
            action=torch.tensor([t.action for t in batch]),
            reward=torch.tensor([t.reward for t in batch]),
            next_obs=torch.stack([t.next_obs for t in batch]),
            done=torch.tensor([t.done for t in batch]),
        )

        # compute loss
        loss = self._compute_loss(batch)

        # backpropagate
        loss.backward()


class DQNAgent(nn.Module):
    """DQN Agent implementation"""

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Create networks
        self.q = DuelingQNetwork(
            num_actions=cfg.act_dim, input_dim=cfg.obs_dim, hidden_dim=cfg.hidden_dim
        )

        self.target_q = DuelingQNetwork(
            num_actions=cfg.act_dim, input_dim=cfg.obs_dim, hidden_dim=cfg.hidden_dim
        )

        # Sync target network initially
        self.sync_target()

    def _compute_loss(self, batch: List[Transition]):
        """Compute standard Double DQN loss

        dqn loss = (r + gamma * max_a Q(s', a') - Q(s, a))**2
        """
        pass

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
        q_values, _ = self.q(obs_tensor)

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

        # Advantage and value heads
        self.advantage = nn.Linear(hidden_dim, num_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass

        Args:
            obs: Batch of obs (batch_size, input_dim)
            hidden_state: Not used, kept for compatibility

        Returns:
            q_values: Q-values for each action
            next_hidden: Always None for MLP
        """
        # Get obs embeddings
        obs_embed = self.obs_embedder(obs)

        # Compute advantage and value streams
        advantage = self.advantage(obs_embed)
        value = self.value(obs_embed)

        # Combine using dueling formula
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values, None
