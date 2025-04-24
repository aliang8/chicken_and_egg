from typing import List

import numpy as np

from chicken_and_egg.utils.data_utils import Transition


class ReplayBuffer:
    """Basic replay buffer implementation"""

    def __init__(self, size: int):
        """Create Replay buffer.

        Args:
            size: Max number of transitions to store in the buffer
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: Transition):
        """Add an transition to the buffer"""
        if self._next_idx >= len(self._storage):
            self._storage.append(transition)
        else:
            self._storage[self._next_idx] = transition
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions

        Args:
            batch_size: How many transitions to sample

        Returns:
            List of sampled transitions
        """
        indices = np.random.randint(len(self._storage), size=batch_size)
        return [self._storage[i] for i in indices]

    def size(self) -> int:
        return len(self._storage)


class SequentialReplayBuffer(ReplayBuffer):
    """Replay buffer that samples sequences of contiguous transitions"""

    def __init__(self, size: int, sequence_length: int = 10, **kwargs):
        """Initialize sequential buffer

        Args:
            size: Max number of sequences to store
            sequence_length: Length of each sequence
        """
        super().__init__(size)
        self._sequence_length = sequence_length
        self._current_sequence = []
        self._first_transition_of_sequence = True

    def add(self, transition: Transition):
        """Add transition to current sequence"""
        if self._first_transition_of_sequence:
            self._first_transition_of_sequence = False
            if self._next_idx >= len(self._storage):
                self._storage.append([])
            self._storage[self._next_idx] = []
            self._current_sequence = self._storage[self._next_idx]

        self._current_sequence.append(transition)

        if transition.done or len(self._current_sequence) >= self._sequence_length:
            self._first_transition_of_sequence = True
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size: int) -> List[List[Transition]]:
        """Sample a batch of sequences

        Args:
            batch_size: Number of sequences to sample

        Returns:
            List of sequences, where each sequence is a list of transitions
        """
        indices = np.random.randint(len(self._storage), size=batch_size)
        sequences = []
        for idx in indices:
            sequence = self._storage[idx]
            if len(sequence) > self._sequence_length:
                start = np.random.randint(0, len(sequence) - self._sequence_length + 1)
                sequence = sequence[start : start + self._sequence_length]
            sequences.append(sequence)
        return sequences

    def size(self) -> int:
        return len(self._storage)
