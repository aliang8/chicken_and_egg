from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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
