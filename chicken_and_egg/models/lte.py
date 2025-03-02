from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig


class TransformerBlock(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.hidden_dim, eps=cfg.layer_norm_epsilon)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_dim,
            cfg.n_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(cfg.hidden_dim, eps=cfg.layer_norm_epsilon)
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
        x, _ = self.attn(x, x, x, attn_mask=attention_mask)
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
        self.wpe = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.hidden_dim, eps=cfg.layer_norm_epsilon)

    def forward(self, input_embeds, position_ids, attention_mask=None):
        position_embeds = self.wpe(position_ids)
        hidden_states = input_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class LTE(nn.Module):
    def __init__(self, cfg: DictConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg

        # Embedding layers
        self.embed_reward = nn.Linear(1, cfg.hidden_dim)
        self.embed_action = nn.Linear(cfg.act_dim, cfg.hidden_dim)

        # Prediction heads
        self.pred_max = nn.Linear(cfg.hidden_dim, cfg.act_dim)
        self.pred_exp = nn.Linear(cfg.hidden_dim, cfg.act_dim)
        self.pred_nonmax = nn.Linear(cfg.hidden_dim, cfg.act_dim)
        self.pred_nonexp = nn.Linear(cfg.hidden_dim, cfg.act_dim)

        # Main transformer model
        self.transformer = TransformerModel(cfg)
        self.ln = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Embed the actions and rewards together as a single token and feed it in

        Args:
            actions: [B, T, A]
            rewards: [B, T, 1]
            position_ids: [B, T]
            attention_mask: Optional [B, T]
        """
        # Embed inputs
        rew_embeds = self.embed_reward(rewards)
        act_embeds = self.embed_action(actions)

        # Combine embeddings
        embeddings = rew_embeds + act_embeds
        embeddings = self.ln(embeddings)

        # Pass through transformer
        output = self.transformer(
            input_embeds=embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        return output
