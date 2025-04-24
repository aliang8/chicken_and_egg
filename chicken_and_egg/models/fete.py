from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from chicken_and_egg.models.base import BaseModel
import transformers
from chicken_and_egg.models.trajectory_gpt2 import GPT2Model


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
        # x, weights = self.attn(x, x, x, key_padding_mask=attention_mask, attn_mask=attention_mask)
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
        self.positional_encodings = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        input_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        position_embeds = self.positional_encodings(timesteps)
        hidden_states = input_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class FETEPolicy(BaseModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg

        # Embedding layers
        self.embed_reward = nn.Linear(1, cfg.hidden_dim)
        self.embed_action = nn.Linear(1, cfg.hidden_dim)
        # self.embed_action = nn.Linear(cfg.act_dim, cfg.hidden_dim)
        self.embed_observation = nn.Linear(cfg.obs_dim, cfg.hidden_dim)
        self.embed_trial_id = nn.Embedding(cfg.num_episodes + 1, cfg.hidden_dim)

        # GPT-style transformer model
        # self.transformer = TransformerModel(cfg)

        self.positional_encodings = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=cfg.hidden_dim,
            n_head=cfg.n_head,
            n_layer=cfg.num_layers,
            dropout=cfg.dropout,
            n_ctx=cfg.max_seq_len,
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.ln = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        timesteps: torch.Tensor,
        episode_ids: Optional[torch.Tensor] = None,
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

        # convert actions to one-hot
        # actions = F.one_hot(actions.long(), num_classes=self.cfg.act_dim).squeeze()
        # act_embeds = self.embed_action(actions.float())
        act_embeds = self.embed_action(actions)

        # Combine embeddings
        embeddings = rew_embeds + act_embeds + obs_embeds

        if episode_ids is not None:
            trial_id_embeds = self.embed_trial_id(episode_ids)
            embeddings = embeddings + trial_id_embeds
        else:
            import ipdb; ipdb.set_trace()

        if timesteps is not None:
            timestep_embeds = self.positional_encodings(timesteps)
            embeddings = embeddings + timestep_embeds
        else:
            import ipdb; ipdb.set_trace()

        embeddings = self.ln(embeddings)

        # # Pass through transformer
        # output = self.transformer(
        #     input_embeds=embeddings,
        #     timesteps=timesteps,
        #     attention_mask=attention_mask,
        # )
        output = self.transformer(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        output = output.last_hidden_state

        return output


class FETE(BaseModel):
    """
    First-Explore Then Exploit policy.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Roll model (behavior policy)
        self.roll_backbone = FETEPolicy(cfg)
        self.roll_explore_head = nn.Linear(cfg.hidden_dim, cfg.env.act_dim)
        self.roll_exploit_head = nn.Linear(cfg.hidden_dim, cfg.env.act_dim)

        # Pred model (successor policy)
        self.pred_backbone = FETEPolicy(cfg)
        self.pred_explore_head = nn.Linear(cfg.hidden_dim, cfg.env.act_dim)
        self.pred_exploit_head = nn.Linear(cfg.hidden_dim, cfg.env.act_dim)

        # Initialize cache
        self.cache_len = (cfg.num_episodes + 1) * (cfg.env.timesteps_per_episode + 1)
        self._init_cache()

    def _init_cache(self):
        """Initialize the cache for autoregressive sampling"""
        self.cache = {
            "observations": torch.zeros(1, self.cache_len, self.cfg.env.obs_dim),
            "rewards": torch.zeros(1, self.cache_len, 1),
            "actions": torch.zeros(1, self.cache_len, 1),
            "timesteps": torch.zeros(1, self.cache_len, dtype=torch.long),
            "episode_ids": torch.zeros(1, self.cache_len, dtype=torch.long),
            "mask": torch.zeros(1, self.cache_len),
        }

    def update_behavior_policy(self):
        # copy weights from pred to roll
        self._copy_params(self.pred_backbone, self.roll_backbone)
        self._copy_params(self.pred_explore_head, self.roll_explore_head)
        self._copy_params(self.pred_exploit_head, self.roll_exploit_head)

    def _copy_params(self, src_policy, dst_policy):
        for src_param, dst_param in zip(src_policy.parameters(), dst_policy.parameters()):
            dst_param.data.copy_(src_param.data)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        episode_ids: Optional[torch.Tensor] = None,
        policy_type: str = "explore_roll",
        **kwargs,
    ):
        # Initialize backbone and head to None
        backbone = None
        head = None

        # Determine which backbone and head to use based on policy_type
        if policy_type == "explore_roll":
            backbone = self.roll_backbone
            head = self.roll_explore_head
        elif policy_type == "explore_pred":
            backbone = self.pred_backbone
            head = self.pred_explore_head
        elif policy_type == "exploit_roll":
            backbone = self.roll_backbone
            head = self.roll_exploit_head
        elif policy_type == "exploit_pred":
            backbone = self.pred_backbone
            head = self.pred_exploit_head
        else:
            raise ValueError(f"Invalid policy_type: {policy_type}")

        # Ensure backbone and head are defined
        if backbone is None or head is None:
            raise ValueError(f"Failed to initialize backbone or head for policy_type: {policy_type}")

        output = backbone(
            observations=observations,
            actions=actions,
            rewards=rewards,
            timesteps=timesteps,
            attention_mask=attention_mask,
            episode_ids=episode_ids,
        )
        output = head(output)
        return output
