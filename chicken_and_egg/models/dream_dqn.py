# Adapted from https://github.com/ezliu/hrl
import collections

import numpy as np
import torch
import utils
from omegaconf import DictConfig
from torch import nn
from torch.nn import utils as torch_utils


class DQNAgent(object):
    def __init__(self, cfg: DictConfig, dqn):
        self._dqn = dqn
        self._updates = 0

        self._losses = collections.deque(maxlen=100)
        self._grad_norms = collections.deque(maxlen=100)

    def update(self, experience):
        """Updates agent on this experience.

        Args:
          experience (Experience): experience to update on.
        """
        self._replay_buffer.add(experience)

        if len(self._replay_buffer) >= self._min_buffer_size:
            if self._updates % self._update_freq == 0:
                experiences = self._replay_buffer.sample(self._batch_size)

                self._optimizer.zero_grad()
                loss = self._dqn.loss(experiences, np.ones(self._batch_size))
                loss.backward()
                self._losses.append(loss.item())

                # clip according to the max allowed grad norm
                grad_norm = torch_utils.clip_grad_norm_(
                    self._dqn.parameters(), self._max_grad_norm, norm_type=2
                )
                self._grad_norms.append(grad_norm)
                self._optimizer.step()

            if self._updates % self._sync_freq == 0:
                self._dqn.sync_target()

        self._updates += 1

    def act(self, state, prev_hidden_state=None, test=False):
        """Given the current state, returns an action.

        Args:
          state (State)

        Returns:
          action (int)
          hidden_state (object)
        """
        return self._dqn.act(state, prev_hidden_state=prev_hidden_state, test=test)

    def set_reward_relabeler(self, reward_relabeler):
        """See DQNPolicy.reward_relabeler."""
        self._dqn.set_reward_relabeler(reward_relabeler)


# TODO(evzliu): Add Policy base class
class DQNPolicy(nn.Module):
    def __init__(self, cfg: DictConfig):
        """DQNPolicy"""
        super().__init__()
        self._Q = DuelingNetwork(cfg.num_actions, cfg.state_embedder_factory())
        self._target_Q = DuelingNetwork(cfg.num_actions, cfg.state_embedder_factory())
        self._reward_relabeler = None

        # Used for generating statistics about the policy
        # Average of max Q values
        self._max_q = collections.deque(maxlen=1000)
        self._min_q = collections.deque(maxlen=1000)
        self._losses = collections.defaultdict(lambda: collections.deque(maxlen=1000))

    def act(self, state, prev_hidden_state=None, test=False):
        """
        Args:
          state (State)
          test (bool): if True, takes on the test epsilon value
          prev_hidden_state (object | None): unused agent state.
          epsilon (float | None): if not None, overrides the epsilon greedy
          schedule with this epsilon value. Mutually exclusive with test
          flag

        Returns:
          int: action
          hidden_state (None)
        """
        del prev_hidden_state

        q_values, hidden_state = self._Q([state], None)
        if test:
            epsilon = self._test_epsilon
        else:
            epsilon = self._epsilon_schedule.step()
        self._max_q.append(torch.max(q_values).item())
        self._min_q.append(torch.min(q_values).item())
        return epsilon_greedy(q_values, epsilon)[0], None

    def loss(self, experiences, weights):
        """Updates parameters from a batch of experiences

        Minimizing the loss:

          (target - Q(s, a))^2

          target = r if done
               r + \gamma * max_a' Q(s', a')

        Args:
          experiences (list[Experience]): batch of experiences, state and
            next_state may be LazyFrames or np.arrays
          weights (list[float]): importance weights on each experience

        Returns:
          loss (torch.tensor): MSE loss on the experiences.
        """
        batch_size = len(experiences)
        states = [e.state for e in experiences]
        actions = torch.tensor([e.action for e in experiences]).long()
        next_states = [e.next_state for e in experiences]
        rewards = torch.tensor([e.reward for e in experiences]).float()

        # (batch_size,) 1 if was not done, otherwise 0
        not_done_mask = torch.tensor([1 - e.done for e in experiences]).byte()
        weights = torch.tensor(weights).float()

        # TODO(evzliu): Could more gracefully incorporate aux_losses
        current_state_q_values, aux_losses = self._Q(states, None)
        if isinstance(aux_losses, dict):
            for name, loss in aux_losses.items():
                self._losses[name].append(loss.detach().cpu().data.numpy())
        current_state_q_values = current_state_q_values.gather(1, actions.unsqueeze(1))

        # DDQN
        best_actions = torch.max(self._Q(next_states, None)[0], 1)[1].unsqueeze(1)
        next_state_q_values = (
            self._target_Q(next_states, None)[0].gather(1, best_actions).squeeze(1)
        )
        targets = rewards + self._gamma * (next_state_q_values * not_done_mask.float())
        targets.detach_()  # Don't backprop through targets

        td_error = current_state_q_values.squeeze() - targets
        loss = torch.mean((td_error**2) * weights)
        self._losses["td_error"].append(loss.detach().cpu().data.numpy())
        aux_loss = 0
        if isinstance(aux_losses, dict):
            aux_loss = sum(aux_losses.values())
        return loss + aux_loss

    def sync_target(self):
        """Syncs the target Q values with the current Q values"""
        self._target_Q.load_state_dict(self._Q.state_dict())

    def set_reward_relabeler(self, reward_relabeler):
        """Sets the reward relabeler when computing the loss.

        Args:
          reward_relabeler (RewardLabeler)

        Raises:
          ValueError: when the reward relabeler has already been set.
        """
        if self._reward_relabeler is not None:
            raise ValueError("Reward relabeler already set.")
        self._reward_relabeler = reward_relabeler


class RecurrentDQNPolicy(DQNPolicy):
    """Implements a DQN policy that uses an RNN on the observations."""

    def loss(self, experiences, weights):
        """Updates recurrent parameters from a batch of sequential experiences

        Minimizing the DQN loss:

          (target - Q(s, a))^2

          target = r if done
               r + \gamma * max_a' Q(s', a')

        Args:
          experiences (list[list[Experience]]): batch of sequences of experiences.
          weights (list[float]): importance weights on each experience

        Returns:
          loss (torch.tensor): MSE loss on the experiences.
        """
        unpadded_experiences = experiences
        experiences, mask = utils.pad(experiences)
        batch_size = len(experiences)
        seq_len = len(experiences[0])

        hidden_states = [seq[0].agent_state for seq in experiences]
        # Include the next states in here to minimize calls to _Q
        states = [[e.state for e in seq] + [seq[-1].next_state] for seq in experiences]
        actions = torch.tensor([e.action for seq in experiences for e in seq]).long()
        next_hidden_states = [seq[0].next_agent_state for seq in experiences]
        next_states = [[e.next_state for e in seq] for seq in experiences]
        rewards = torch.tensor([e.reward for seq in experiences for e in seq]).float()

        # TODO(evzliu): Could more gracefully handle this by passing a
        # TrajectoryExperience object to label_rewards to take TrajectoryExperience
        # Relabel the rewards on the fly
        if self._reward_relabeler is not None:
            trajectories = [seq[0].trajectory for seq in experiences]
            # (batch_size, max_seq_len)
            indices = torch.tensor(
                [[e.index for e in seq] for seq in experiences]
            ).long()

            # (batch_size * max_trajectory_len)
            rewards = (
                self._reward_relabeler.label_rewards(trajectories)[0]
                .gather(-1, indices)
                .reshape(-1)
            )

        # (batch_size,) 1 if was not done, otherwise 0
        not_done_mask = ~(
            torch.tensor([e.done for seq in experiences for e in seq]).bool()
        )
        weights = torch.tensor(weights).float()

        # (batch_size, seq_len + 1, actions)
        q_values, _ = self._Q(states, hidden_states)
        current_q_values = q_values[:, :-1, :]
        current_q_values = current_q_values.reshape(batch_size * seq_len, -1)
        # (batch_size * seq_len, 1)
        current_state_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        # TODO(evzliu): Could more gracefully incorporate aux_losses
        aux_losses = {}
        if hasattr(self._Q._state_embedder, "aux_loss"):
            aux_losses = self._Q._state_embedder.aux_loss(unpadded_experiences)
            if isinstance(aux_losses, dict):
                for name, loss in aux_losses.items():
                    self._losses[name].append(loss.detach().cpu().data.numpy())

        # DDQN
        next_q_values = q_values[:, 1:, :]
        # (batch_size * seq_len, actions)
        next_q_values = next_q_values.reshape(batch_size * seq_len, -1)
        best_actions = torch.max(next_q_values, 1)[1].unsqueeze(1)
        # Using the same hidden states for target
        target_q_values, _ = self._target_Q(next_states, next_hidden_states)
        target_q_values = target_q_values.reshape(batch_size * seq_len, -1)
        next_state_q_values = target_q_values.gather(1, best_actions).squeeze(1)
        targets = rewards + self._gamma * (next_state_q_values * not_done_mask.float())
        targets.detach_()  # Don't backprop through targets

        td_error = current_state_q_values.squeeze() - targets
        weights = weights.unsqueeze(1) * mask.float()
        loss = (td_error**2).reshape(batch_size, seq_len) * weights
        loss = loss.sum() / mask.sum()  # masked mean
        return loss + sum(aux_losses.values())

    def act(self, state, prev_hidden_state=None, test=False):
        """
        Args:
          state (State)
          test (bool): if True, takes on the test epsilon value
          prev_hidden_state (object | None): unused agent state.
          epsilon (float | None): if not None, overrides the epsilon greedy
          schedule with this epsilon value. Mutually exclusive with test
          flag

        Returns:
          int: action
          hidden_state (None)
        """
        q_values, hidden_state = self._Q([[state]], prev_hidden_state)
        if test:
            epsilon = self._test_epsilon
        else:
            epsilon = self._epsilon_schedule.step()
        self._max_q.append(torch.max(q_values).item())
        self._min_q.append(torch.min(q_values).item())
        return epsilon_greedy(q_values, epsilon)[0], hidden_state


class DQN(nn.Module):
    """Implements the Q-function."""

    def __init__(self, num_actions, state_embedder):
        """
        Args:
          num_actions (int): the number of possible actions at each state
          state_embedder (StateEmbedder): the state embedder to use
        """
        super(DQN, self).__init__()
        self._state_embedder = state_embedder
        self._q_values = nn.Linear(self._state_embedder.embed_dim, num_actions)

    def forward(self, states, hidden_states=None):
        """Returns Q-values for each of the states.

        Args:
          states (FloatTensor): shape (batch_size, 84, 84, 4)
          hidden_states (object | None): hidden state returned by previous call to
            forward. Must be called on constiguous states.

        Returns:
          FloatTensor: (batch_size, num_actions)
          hidden_state (object)
        """
        state_embed, hidden_state = self._state_embedder(states, hidden_states)
        return self._q_values(state_embed), hidden_state


class DuelingNetwork(DQN):
    """Implements the following Q-network:

    Q(s, a) = V(s) + A(s, a) - avg_a' A(s, a')
    """

    def __init__(self, num_actions, state_embedder):
        super(DuelingNetwork, self).__init__(num_actions, state_embedder)
        self._V = nn.Linear(self._state_embedder.embed_dim, 1)
        self._A = nn.Linear(self._state_embedder.embed_dim, num_actions)

    def forward(self, states, hidden_states=None):
        state_embedding, hidden_state = self._state_embedder(states, hidden_states)
        V = self._V(state_embedding)
        advantage = self._A(state_embedding)
        mean_advantage = torch.mean(advantage)
        return V + advantage - mean_advantage, hidden_state


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
