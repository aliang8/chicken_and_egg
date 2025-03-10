from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from chicken_and_egg.models.base import BaseModel


class TransformerBlock(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_dim,
            cfg.n_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(cfg.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 4 * cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.hidden_dim, cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self attention
        residual = x
        x = self.ln_1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=attention_mask)
        x = residual + x

        # MLP
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class TransformerModel(nn.Module):
    """A PyTorch version of the modified GPT2 model"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Standard transformer encoder components
        self.wpe = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, input_embeds, timesteps, attention_mask=None):
        position_embeds = self.wpe(timesteps)
        hidden_states = input_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class FETEPolicy(BaseModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Embedding layers
        self.embed_reward = nn.Linear(1, cfg.hidden_dim)
        self.embed_action = nn.Linear(1, cfg.hidden_dim)
        self.embed_observation = nn.Linear(cfg.obs_dim, cfg.hidden_dim)

        # GPT-style transformer model
        self.transformer = TransformerModel(cfg)
        self.ln = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Embed the actions and rewards together as a single token and feed it into the transformer.

        Args:
            observations: [B, T, O]
            actions: [B, T, A]
            rewards: [B, T, 1]
            timesteps: [B, T]
            attention_mask: Optional [B, T]
        """
        # Embed inputs
        obs_embeds = self.embed_observation(observations)
        rew_embeds = self.embed_reward(rewards)
        act_embeds = self.embed_action(actions)

        # Combine embeddings
        embeddings = rew_embeds + act_embeds + obs_embeds
        embeddings = self.ln(embeddings)

        # Pass through transformer
        output = self.transformer(
            input_embeds=embeddings,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )

        return output


class FETE(BaseModel):
    """
    First-Explore Then Exploit policy.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.policy_backbone = FETEPolicy(cfg)

        # Explore and exploit share the same backbone but different output heads
        self.explore_head = nn.Linear(cfg.hidden_dim, cfg.act_dim)
        self.exploit_head = nn.Linear(cfg.hidden_dim, cfg.act_dim)

        self.successor_backbone = FETEPolicy(cfg)

        self.successor_explore_head = nn.Linear(cfg.hidden_dim, cfg.act_dim)
        self.successor_exploit_head = nn.Linear(cfg.hidden_dim, cfg.act_dim)

    def update_behavior_policy(self):
        # copy weights from successor to behavior
        self._copy_params(self.successor_backbone, self.policy_backbone)
        self._copy_params(self.successor_explore_head, self.explore_head)
        self._copy_params(self.successor_exploit_head, self.exploit_head)

    def _copy_params(self, src_policy, dst_policy):
        for param, successor_param in zip(
            src_policy.parameters(),
            dst_policy.parameters(),
        ):
            param.data.copy_(successor_param.data)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        policy_type: str = "explore_behavior",
    ):
        if policy_type == "explore_behavior":
            head = self.explore_head
        elif policy_type == "explore_successor":
            head = self.successor_explore_head
        elif policy_type == "exploit_behavior":
            head = self.exploit_head
        elif policy_type == "exploit_successor":
            head = self.successor_exploit_head

        if "behavior" in policy_type:
            backbone = self.policy_backbone
        elif "successor" in policy_type:
            backbone = self.successor_backbone

        output = backbone(observations, actions, rewards, timesteps, attention_mask)
        output = head(output)
        return output
