import einops
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Categorical

# Import environment and model after setting path
from chicken_and_egg.envs.bandit import Bandit, MeanBandit
from chicken_and_egg.models.lte import LTE
from chicken_and_egg.utils.logger import log


class FETETrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Device: {self.device}")

        # Set random seeds
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        # Initialize environment
        self.env = self._setup_environment()
        log(f"Environment: {self.env}")

        # Initialize model and optimizer
        self.model = LTE(self.cfg.model).to(self.device)
        log(f"Model: {self.model}")

        self.num_training_steps = 1000 * self.cfg.env.train_len

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.model.lr,
            weight_decay=cfg.model.weight_decay,
        )
        self.scheduler = self._setup_scheduler()

        # Calculate maximum reward for environment
        self.max_reward = self._calculate_max_reward()
        log(f"Max reward: {self.max_reward}")
        # Setup wandb
        self.run_name = (
            f"n{self.cfg.env.act_dim}_p{self.cfg.env.seq_len}_b{self.cfg.data.batch_size}_"
            f"{self.cfg.env.bandit_type}_seed{self.cfg.seed}"
        )
        log(f"Run name: {self.run_name}")

        if self.cfg.train.wandb:
            self._setup_wandb()

    def _setup_environment(self):
        if self.cfg.env.bandit_type == "normal":
            return Bandit(n=self.cfg.env.act_dim, deterministic=False, noise_scale=0.5)
        elif self.cfg.env.bandit_type == "mean":
            return MeanBandit(n=self.cfg.env.act_dim, minval=0.5)
        elif self.cfg.env.bandit_type == "control":
            return MeanBandit(n=self.cfg.env.act_dim, minval=0)

    def _setup_scheduler(self):
        warmup_steps = int(self.cfg.model.warmup_fraction * self.num_training_steps)
        return optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: min(
                (step + 1) / warmup_steps,
                max(
                    0.0,
                    (self.num_training_steps - step)
                    / (self.num_training_steps - warmup_steps),
                ),
            ),
        )

    def _calculate_max_reward(self):
        if self.cfg.env.bandit_type == "normal":
            return (
                torch.normal(0, 1, size=(10000, self.cfg.env.act_dim))
                .max(dim=1)[0]
                .mean()
                .item()
            )
        else:
            vals = torch.normal(0, 1, size=(10000, self.cfg.env.act_dim))
            vals[:, 0] = 0.5 if self.cfg.env.bandit_type == "mean" else 0
            return vals.max(dim=1)[0].mean().item()

    def _setup_wandb(self):
        wandb.init(
            project="bandit-first-explore",
            name=self.run_name,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )

    def batch_step(self, states, actions):
        return [self.env.step(s, a) for s, a in zip(states, actions)]

    def batch_mset(self, B):
        return [self.env.meta_reset() for _ in range(B)]

    def reward_sequence(self, states, actions):
        rewards = []
        for t in range(actions.shape[1]):
            states = self.batch_step(states, actions[:, t])
            rewards.append([s["reward"] for s in states])
        return torch.tensor(rewards).T.to(self.device)

    def max_in_seq(self, values: torch.Tensor) -> torch.Tensor:
        running_max = torch.zeros_like(values)
        running_max[:, 0] = values[:, 0]
        for t in range(1, values.shape[1]):
            running_max[:, t] = torch.maximum(running_max[:, t - 1], values[:, t])
        return running_max

    def tokenize(
        self, actions_batch: torch.Tensor, rewards_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_tokens = torch.zeros(
            actions_batch.shape[0], actions_batch.shape[1], self.cfg.env.act_dim
        ).to(self.device)
        action_tokens.scatter_(2, actions_batch.unsqueeze(-1), 1)
        reward_tokens = rewards_batch.unsqueeze(-1).to(self.device)
        return action_tokens, reward_tokens

    def epsilon_samp(self, logits: torch.Tensor, epsilon: float) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / probs.shape[-1]
        mixed_probs = (1 - epsilon) * probs + epsilon * uniform
        return torch.log(mixed_probs)

    def exploit(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        running_max: torch.Tensor,
        argmax: bool = False,
        greater: bool = False,
        epsilon: float = 0,
    ):
        # [B, T, A], [B, T, 1]
        action_tokens, reward_tokens = self.tokenize(actions, rewards)
        B = actions.shape[0]

        # Add initial zero tokens
        action_tokens = torch.cat(
            [torch.zeros(B, 1, self.cfg.env.act_dim).to(self.device), action_tokens],
            dim=1,
        )
        reward_tokens = torch.cat(
            [torch.zeros(B, 1, 1).to(self.device), reward_tokens], dim=1
        )
        T = actions.shape[1]

        # [B, T]
        timesteps = torch.arange(T + 1).repeat(B, 1).to(self.device)

        hidden_state = self.model(action_tokens, reward_tokens, position_ids=timesteps)
        max_logits = self.model.pred_max(hidden_state)
        nonmax_logits = self.model.pred_nonmax(hidden_state)

        sample_logits = (
            self.epsilon_samp(max_logits, epsilon)
            if epsilon > 0
            else max_logits.detach()
        )
        m_actions = (
            torch.argmax(max_logits, dim=-1)
            if argmax
            else Categorical(logits=sample_logits).sample()
        )

        m_rewards = self.reward_sequence(states, m_actions)
        running_max = torch.cat(
            [torch.full((B, 1), float("-inf")).to(self.device), running_max],
            dim=1,
        )

        action_preds = torch.where(
            (m_rewards >= running_max).unsqueeze(-1)
            if not greater
            else (m_rewards > running_max).unsqueeze(-1),
            max_logits,
            nonmax_logits,
        )

        pred = action_preds + sample_logits
        pred = einops.rearrange(pred, "B T A -> (B T) A")
        m_actions = einops.rearrange(m_actions, "B T -> (B T)")
        loss = nn.CrossEntropyLoss()(pred, m_actions)
        return m_rewards, loss

    def train_step(self, batch_size: int):
        self.model.train()
        self.optimizer.zero_grad()
        states = self.batch_mset(batch_size)
        actions = torch.randint(
            0, self.cfg.env.act_dim, (batch_size, self.cfg.env.seq_len - 1)
        ).to(self.device)
        rewards = self.reward_sequence(states, actions)
        running_max = self.max_in_seq(rewards)

        _, loss = self.exploit(
            states, actions, rewards, running_max, epsilon=self.cfg.train.epsilon
        )
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def eval_step(self, batch_size: int):
        self.model.eval()
        with torch.no_grad():
            states = self.batch_mset(batch_size)
            actions = torch.randint(
                0, self.cfg.env.act_dim, (batch_size, self.cfg.env.seq_len - 1)
            ).to(self.device)
            rewards = self.reward_sequence(states, actions)
            running_max = self.max_in_seq(rewards)
            _, loss = self.exploit(
                states, actions, rewards, running_max, epsilon=self.cfg.train.epsilon
            )
            return loss.item()

    def train(self):
        for epoch in tqdm.tqdm(
            range(self.cfg.train.num_epochs),
            desc="Training",
            total=self.cfg.train.num_epochs,
        ):
            train_loss = self.train_step(self.cfg.data.batch_size)

            if epoch % self.cfg.train.eval_interval == 0:
                eval_loss = self.eval_step(self.cfg.data.batch_size)

                if self.cfg.train.wandb:
                    wandb.log(
                        {
                            "train/loss": train_loss,
                            "eval/loss": eval_loss,
                            "epoch": epoch,
                        }
                    )
                log(
                    f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Eval Loss = {eval_loss:.4f}"
                )
