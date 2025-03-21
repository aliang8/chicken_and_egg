from typing import Any, Dict, List, Optional

import torch


class Experience:
    """Class to store a single step of experience"""

    def __init__(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        info: Dict,
        hidden_state: Optional[torch.Tensor] = None,
        next_hidden_state: Optional[torch.Tensor] = None,
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info
        self.hidden_state = hidden_state
        self.next_hidden_state = next_hidden_state


class TrajectoryExperience:
    """Wrapper class to store experience with trajectory context"""

    def __init__(
        self, experience: Experience, trajectory: List[Experience], index: int
    ):
        self.experience = experience
        self.trajectory = trajectory
        self.index = index
