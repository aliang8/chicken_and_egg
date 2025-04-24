from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class Transition:
    obs: Any
    action: int
    reward: float
    next_obs: Any
    done: bool
    info: Dict = field(default_factory=dict)
    hidden_state: Optional[torch.Tensor] = None
    trajectory: Optional[List[Any]] = None
    index: Optional[int] = None
    env_id: Optional[np.ndarray] = None
