# Adapted from https://github.com/ezliu/hrl
import collections
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from chicken_and_egg.utils.data_utils import TrajectoryExperience
from chicken_and_egg.utils.replay_buffer import ReplayBuffer, SequentialReplayBuffer


class DQNAgent:
    """DQN Agent implementation"""

    def __init__(self, cfg: DictConfig, env):
        self.cfg = cfg
        self.env = env

        # Create networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DuelingNetwork(
            num_actions=env.action_space.n,
            input_dim=env.observation_space.shape[0],  # Assuming flat observations
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        self.target_network = DuelingNetwork(
            num_actions=env.action_space.n,
            input_dim=env.observation_space.shape[0],
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)

        # Sync target network initially
        self.sync_target()

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=cfg.learning_rate
        )

        # Initialize tracking variables
        self.updates = 0
        self.epsilon = cfg.initial_epsilon
        self.losses = collections.deque(maxlen=100)
        self.grad_norms = collections.deque(maxlen=100)

        # Initialize replay buffer
        self.replay_buffer = (
            SequentialReplayBuffer(cfg.buffer.buffer_size, cfg.buffer.sequence_length)
            if cfg.buffer.type == "sequential"
            else ReplayBuffer(cfg.buffer.buffer_size)
        )

    def update(self, experience: TrajectoryExperience):
        """Update agent from experience

        Args:
            experience: TrajectoryExperience containing the step and context
        """
        # Add to replay buffer
        self.replay_buffer.add(experience)

        # Only update if buffer has enough samples
        if len(self.replay_buffer) < self.cfg.min_buffer_size:
            return

        # Only update every n steps
        if self.updates % self.cfg.update_freq != 0:
            self.updates += 1
            return

        # Sample batch of experiences
        experiences = self.replay_buffer.sample(self.cfg.batch_size)

        # Handle both sequential and non-sequential buffers
        if isinstance(self.replay_buffer, SequentialReplayBuffer):
            # Flatten sequences for loss computation
            flat_experiences = []
            for sequence in experiences:
                flat_experiences.extend(sequence)
            loss = self._compute_loss(flat_experiences)
        else:
            loss = self._compute_loss(experiences)

        # Compute gradients and update
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        grad_norm = nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.cfg.max_grad_norm
        )

        self.optimizer.step()

        # Track metrics
        self.losses.append(loss.item())
        self.grad_norms.append(grad_norm)

        # Periodically sync target network
        if self.updates % self.cfg.target_update_freq == 0:
            self.sync_target()

        self.updates += 1

    def act(
        self,
        state: np.ndarray,
        hidden_state: Optional[torch.Tensor] = None,
        test: bool = False,
    ) -> Tuple[int, None]:
        """Select action using epsilon-greedy policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values, _ = self.q_network(state_tensor)

        # Use smaller epsilon during testing
        epsilon = self.cfg.test_epsilon if test else self.epsilon

        # Epsilon-greedy action selection
        if np.random.random() > epsilon:
            action = q_values.argmax(dim=1).item()
        else:
            action = np.random.randint(0, self.env.action_space.n)

        return action, None

    def _compute_loss(self, experiences: List[TrajectoryExperience]) -> torch.Tensor:
        """Compute the Q-learning loss on a batch of experiences"""
        # Move data to device
        states = torch.FloatTensor([e.experience.state for e in experiences]).to(
            self.device
        )
        actions = (
            torch.tensor([e.experience.action for e in experiences])
            .long()
            .to(self.device)
        )
        rewards = (
            torch.tensor([e.experience.reward for e in experiences])
            .float()
            .to(self.device)
        )
        next_states = torch.FloatTensor(
            [e.experience.next_state for e in experiences]
        ).to(self.device)
        dones = (
            torch.tensor([e.experience.done for e in experiences])
            .float()
            .to(self.device)
        )

        # Get current Q values
        current_q, _ = self.q_network(states)
        current_q = current_q.gather(1, actions.unsqueeze(1))

        # Get next Q values using target network
        with torch.no_grad():
            # Double Q-learning
            next_actions = self.q_network(next_states)[0].argmax(dim=1)
            next_q = self.target_network(next_states)[0]
            next_q = next_q.gather(1, next_actions.unsqueeze(1)).squeeze()

        # Compute targets
        targets = rewards + (1 - dones) * self.cfg.gamma * next_q

        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), targets)

        return loss

    def sync_target(self):
        """Sync target network with current Q network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def set_reward_relabeler(self, reward_relabeler):
        """Set the reward relabeling function"""
        self.reward_relabeler = reward_relabeler

    @property
    def stats(self) -> Dict[str, float]:
        """Return current training statistics"""
        return {
            "loss": np.mean(self.losses) if self.losses else None,
            "grad_norm": np.mean(self.grad_norms) if self.grad_norms else None,
            "epsilon": self.epsilon,
        }


class DuelingNetwork(nn.Module):
    """Dueling DQN Network Architecture with MLP state embedder"""

    def __init__(self, num_actions: int, input_dim: int, hidden_dim: int = 64):
        super().__init__()

        # Simple MLP state embedder
        self.state_embedder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )

        # Advantage and value streams
        self.advantage = nn.Linear(hidden_dim, num_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(
        self, states: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass

        Args:
            states: Batch of states (batch_size, input_dim)
            hidden_state: Not used, kept for compatibility

        Returns:
            q_values: Q-values for each action
            next_hidden: Always None for MLP
        """
        # Get state embeddings
        state_embed = self.state_embedder(states)

        # Compute advantage and value streams
        advantage = self.advantage(state_embed)
        value = self.value(state_embed)

        # Combine using dueling formula
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values, None


def epsilon_greedy(q_values, epsilon):
    """Returns the index of the highest q value with prob 1 - epsilon,
    otherwise uniformly at random with prob epsilon.

    Args:
      q_values (Variable[FloatTensor]): (batch_size, num_actions)
      epsilon (float)

    Returns:
      list[int]: actions
    """
    batch_size, num_actions = q_values.size()
    _, max_indices = torch.max(q_values, 1)
    max_indices = max_indices.cpu().data.numpy()
    actions = []
    for i in range(batch_size):
        if np.random.random() > epsilon:
            actions.append(max_indices[i])
        else:
            actions.append(np.random.randint(0, num_actions))
    return actions
