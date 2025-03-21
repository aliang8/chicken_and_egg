from typing import List

import numpy as np
from omegaconf import DictConfig

from chicken_and_egg.utils.data_utils import TrajectoryExperience


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

    @classmethod
    def from_config(cls, cfg: DictConfig):
        """Create buffer from config"""
        buffer_type = cfg.type
        if buffer_type == "vanilla":
            return cls(cfg.buffer_size)
        elif buffer_type == "sequential":
            return SequentialReplayBuffer(
                size=cfg.buffer_size, sequence_length=cfg.sequence_length
            )
        else:
            raise ValueError(f"Unsupported buffer type: {buffer_type}")

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, experience: TrajectoryExperience):
        """Add an experience to the buffer"""
        if self._next_idx >= len(self._storage):
            self._storage.append(experience)
        else:
            self._storage[self._next_idx] = experience
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size: int) -> List[TrajectoryExperience]:
        """Sample a batch of experiences

        Args:
            batch_size: How many transitions to sample

        Returns:
            List of sampled experiences
        """
        indices = np.random.randint(len(self._storage), size=batch_size)
        return [self._storage[i] for i in indices]


class SequentialReplayBuffer(ReplayBuffer):
    """Replay buffer that samples sequences of contiguous experiences"""

    def __init__(self, size: int, sequence_length: int = 10):
        """Initialize sequential buffer

        Args:
            size: Max number of sequences to store
            sequence_length: Length of each sequence
        """
        super().__init__(size)
        self._sequence_length = sequence_length
        self._current_sequence = []
        self._first_experience_of_sequence = True

    def add(self, experience: TrajectoryExperience):
        """Add experience to current sequence"""
        if self._first_experience_of_sequence:
            self._first_experience_of_sequence = False
            if self._next_idx >= len(self._storage):
                self._storage.append([])
            self._storage[self._next_idx] = []
            self._current_sequence = self._storage[self._next_idx]

        self._current_sequence.append(experience)

        if (
            experience.experience.done
            or len(self._current_sequence) >= self._sequence_length
        ):
            self._first_experience_of_sequence = True
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size: int) -> List[List[TrajectoryExperience]]:
        """Sample a batch of sequences

        Args:
            batch_size: Number of sequences to sample

        Returns:
            List of sequences, where each sequence is a list of experiences
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
