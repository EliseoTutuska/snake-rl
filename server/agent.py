"""小 DQN：用棋盘网格学 Q(s, a)，打破 11 维特征的状态混叠。

网络
----
Conv(4→16, k=3, pad=1) → ReLU → Conv(16→32, k=3, pad=1) → ReLU
→ Conv(32→32, k=3, pad=1) → Flatten → Linear→128→3。
观测网格已旋成「朝上」，因此三个 Q 对应画面上的直行/右转/左转。

学习
----
优先经验回放 + 目标网络 + 掩码 Double DQN + Huber TD 损失。
没有 GAE / PPO。设备固定 CPU。

对外接口与旧 QAgent 对齐：act / learn / reset / save / load。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from server.env import OBS_CHANNELS

_ILLEGAL_Q = -1e9
_PRI_EPS = 1e-6


class SnakeDQN(nn.Module):
    """输入 (B,4,H,W) 朝上网格，输出 3 个相对动作的 Q。"""

    def __init__(self, grid_size: int, channels: int = OBS_CHANNELS) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, grid_size, grid_size)
            conv_dim = int(self.conv(dummy).numel())
        self.fc = nn.Sequential(
            nn.Linear(conv_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(grid).flatten(1))


class DqnAgent:
    """ε-greedy DQN。epsilon 由 Trainer 传入，本类不衰减。"""

    def __init__(
        self,
        grid_size: int = 8,
        gamma: float = 0.95,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        warmup: int = 1_000,
        learn_every: int = 4,
        target_every: int = 500,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 80_000,
        device: str = "cpu",
    ) -> None:
        self.grid_size = grid_size
        self.channels = OBS_CHANNELS
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.warmup = warmup
        self.learn_every = learn_every
        self.target_every = target_every
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.device = torch.device("cpu")
        self.device_name = device
        self.online = SnakeDQN(grid_size).to(self.device)
        self.target = SnakeDQN(grid_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self._data: list = [None] * buffer_size
        self._priorities = np.zeros(buffer_size, dtype=np.float64)
        self._size = 0
        self._next = 0
        self._max_priority = 1.0
        self.learn_steps = 0
        self.last_loss = 0.0

    @property
    def replay_size(self) -> int:
        return self._size

    def act(
        self,
        observation: list[float],
        epsilon: float,
        mask: list[bool] | None = None,
    ) -> int:
        """ε-greedy。mask[i]=False 表示该相对动作一步就会死，探索和贪心都跳过。"""
        legal = [i for i, ok in enumerate(mask or [True, True, True]) if ok]
        if not legal:
            return 0
        if random.random() < epsilon:
            return random.choice(legal)
        grid = self._split_obs([observation])
        self.online.eval()
        with torch.no_grad():
            q = self.online(grid)[0].clone()
            self._mask_q_row(q, mask)
            return int(torch.argmax(q).item())

    def learn(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
        mask: list[bool] | None = None,
        next_mask: list[bool] | None = None,
    ) -> float:
        mask = list(mask) if mask is not None else [True, True, True]
        next_mask = list(next_mask) if next_mask is not None else [True, True, True]
        if done:
            next_mask = [True, True, True]
        self._append((state, action, float(reward), next_state, bool(done), mask, next_mask))
        if self._size < self.warmup:
            return self.last_loss
        self.learn_steps += 1
        if self.learn_steps % self.learn_every != 0:
            return self.last_loss
        loss = self._optimize()
        if self.learn_steps % self.target_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.last_loss = loss
        return loss

    def reset(self) -> None:
        """重新随机初始化网络并清空回放池。"""
        self.online = SnakeDQN(self.grid_size).to(self.device)
        self.target = SnakeDQN(self.grid_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.optimizer.defaults["lr"])
        self._data = [None] * self.buffer_size
        self._priorities = np.zeros(self.buffer_size, dtype=np.float64)
        self._size = 0
        self._next = 0
        self._max_priority = 1.0
        self.learn_steps = 0
        self.last_loss = 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "grid_size": self.grid_size,
                "channels": self.channels,
                "gamma": self.gamma,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
            },
            path,
        )

    def load(self, path: Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=True)
        if int(data["grid_size"]) != self.grid_size:
            raise ValueError(
                f"权重棋盘 {data['grid_size']} 与当前 {self.grid_size} 不一致"
            )
        channels = int(data.get("channels", 3))
        if channels != self.channels:
            raise ValueError("旧 3 通道权重，请重置大脑")
        self.online.load_state_dict(data["online"])
        self.target.load_state_dict(data["target"])
        self.gamma = float(data.get("gamma", self.gamma))
        self.online.eval()
        self.target.eval()

    def _append(
        self,
        transition: tuple[list[float], int, float, list[float], bool, list[bool], list[bool]],
    ) -> None:
        self._data[self._next] = transition
        self._priorities[self._next] = self._max_priority
        self._next = (self._next + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def _beta(self) -> float:
        frac = min(1.0, self.learn_steps / max(1, self.per_beta_frames))
        return self.per_beta_start + (1.0 - self.per_beta_start) * frac

    def _optimize(self) -> float:
        indices, is_weights = self._sample_indices()
        batch = [self._data[i] for i in indices]
        states, actions, rewards, next_states, dones, _masks, next_masks = zip(*batch)
        grid = self._split_obs(list(states))
        next_grid = self._split_obs(list(next_states))
        action_t = torch.tensor(actions, dtype=torch.long)
        reward_t = torch.tensor(rewards, dtype=torch.float32)
        done_t = torch.tensor(dones, dtype=torch.float32)
        weight_t = torch.tensor(is_weights, dtype=torch.float32)

        self.online.train()
        q = self.online(grid).gather(1, action_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN：在线网络选动作（先掩非法），目标网络估价值。
            next_q_online = self.online(next_grid).clone()
            self._mask_q_batch(next_q_online, next_masks)
            next_act = next_q_online.argmax(1)
            next_q = self.target(next_grid).gather(1, next_act.unsqueeze(1)).squeeze(1)
            target = reward_t + self.gamma * next_q * (1.0 - done_t)
        td = q - target
        loss = (weight_t * F.smooth_l1_loss(q, target, reduction="none")).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        pri = td.detach().abs().cpu().numpy() + _PRI_EPS
        self._priorities[indices] = pri
        self._max_priority = max(self._max_priority, float(pri.max()))
        return float(loss.item())

    def _sample_indices(self) -> tuple[np.ndarray, np.ndarray]:
        pri = self._priorities[: self._size]
        probs = np.power(pri, self.per_alpha)
        total = float(probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            probs = np.full(self._size, 1.0 / self._size)
        else:
            probs = probs / total
        replace = self._size < self.batch_size
        indices = np.random.choice(self._size, size=self.batch_size, replace=replace, p=probs)
        beta = self._beta()
        weights = np.power(self._size * probs[indices], -beta)
        max_w = float(weights.max())
        if max_w > 0:
            weights = weights / max_w
        return indices, weights

    @staticmethod
    def _mask_q_row(q: torch.Tensor, mask: list[bool] | None) -> None:
        if mask is None:
            return
        if not any(mask) or all(mask):
            return
        for i, ok in enumerate(mask):
            if not ok:
                q[i] = _ILLEGAL_Q

    @staticmethod
    def _mask_q_batch(q: torch.Tensor, masks: tuple[list[bool], ...]) -> None:
        for i, mask in enumerate(masks):
            if not any(mask) or all(mask):
                continue
            for a, ok in enumerate(mask):
                if not ok:
                    q[i, a] = _ILLEGAL_Q

    def _split_obs(self, batch: list[list[float]]) -> torch.Tensor:
        """把扁平观测还原成 (B,4,H,W)。"""
        n = self.grid_size
        t = torch.tensor(batch, dtype=torch.float32)
        return t.reshape(-1, self.channels, n, n)
