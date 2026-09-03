"""训练循环与观看循环。都在独立线程里跑，避免堵住 FastAPI。

Trainer
-------
- 拥有自己的 SnakeEnv（8×8）和 DqnAgent。
- start() 后在内存里尽快 step()，**不经过 HTTP**。
- 约每 80ms 或每局结束时通过 on_update 推一次 status。
- 选动作时用 legal_actions 掩掉一步就死、摸不到尾巴的方向。
- 蛇长 >= 12 时探索率强制为 0；>= 20 时改用跟尾启发式，转移仍写入回放。

Watcher
-------
- 复用同一套 DQN 权重，环境也必须是 8×8（卷积输入尺寸固定）。
- 蛇长 < 20 时 act(..., epsilon=0)；否则跟尾启发式。约 8Hz。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import torch

from server.agent import DqnAgent
from server.env import HEURISTIC_MIN_LENGTH, SnakeEnv
from server.rewards import RewardConfig

MODEL_PATH = Path(__file__).resolve().parent / "models" / "dqn.pt"

UpdateFn = Callable[[dict], None]


class Trainer:
    """进程内 DQN 循环：step → 写入 replay / 优化 →（局终则 reset 并衰减 ε）。"""

    def __init__(
        self,
        grid_size: int = 8,
        seed: int | None = None,
        rewards: RewardConfig | None = None,
        alpha: float = 1e-3,
        gamma: float = 0.97,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.998,
        on_update: UpdateFn | None = None,
        lock: threading.Lock | None = None,
    ) -> None:
        """alpha 传给 Adam 学习率。探索率前 2500 局线性衰减。"""
        self.grid_size = grid_size
        self.seed = seed
        self.rewards = rewards or RewardConfig()
        self.env = SnakeEnv(grid_size=grid_size, seed=seed, rewards=self.rewards)
        self.agent = DqnAgent(grid_size=grid_size, gamma=gamma, lr=alpha)
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episode = 0
        self.total_steps = 0
        self.scores: list[int] = []
        self.last_td_error = 0.0
        self.deaths = {"wall": 0, "self": 0, "timeout": 0}
        self.on_update = on_update
        self.lock = lock or threading.Lock()
        self._obs = self.env.reset()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._push_interval = 0.08

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def step(self) -> dict:
        with self.lock:
            mask = self.env.legal_actions()
            if len(self.env.snake) >= HEURISTIC_MIN_LENGTH:
                action = self.env.heuristic_action(mask)
            else:
                eps = 0.0 if len(self.env.snake) >= 12 else self.epsilon
                action = self.agent.act(self._obs, eps, mask)
            result = self.env.step(action)
            next_mask = [True, True, True] if result.done else self.env.legal_actions()
            self.last_td_error = self.agent.learn(
                self._obs,
                action,
                result.reward,
                result.observation,
                result.done,
                mask,
                next_mask,
            )
            self._obs = result.observation
            self.total_steps += 1
            done = result.done
            score = result.info.score
            if done:
                reason = result.info.reason
                if reason in self.deaths:
                    self.deaths[reason] += 1
                self.episode += 1
                self.scores.append(score)
                if len(self.scores) > 400:
                    self.scores = self.scores[-400:]
                # 线性降到 min，避免表格 Q 那种「几百局就贪心」、网络还没学够。
                frac = min(1.0, self.episode / 2500)
                self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_min) * frac
                self._obs = self.env.reset()
        return {"done": done, "score": score, "action": action}

    def run_episodes(self, n: int) -> list[int]:
        got: list[int] = []
        while len(got) < n:
            result = self.step()
            if result["done"]:
                got.append(result["score"])
        return got

    def mean_score(self, last: int = 50) -> float:
        if not self.scores:
            return 0.0
        slice_ = self.scores[-last:]
        return sum(slice_) / len(slice_)

    def status(self) -> dict:
        return {
            "channel": "train",
            "running": self.running,
            "episode": self.episode,
            "epsilon": self.epsilon,
            "meanScore": self.mean_score(),
            "totalSteps": self.total_steps,
            "scores": list(self.scores[-200:]),
            "qSize": self.agent.replay_size,
            "lastTdError": self.last_td_error,
            "device": "cpu",
            "deaths": dict(self.deaths),
            "snapshot": self.env.snapshot(),
        }

    def reset_learning(self) -> None:
        was_running = self.running
        self.stop()
        with self.lock:
            self.agent.reset()
            self.epsilon = self.epsilon_start
            self.episode = 0
            self.total_steps = 0
            self.scores = []
            self.last_td_error = 0.0
            self.deaths = {"wall": 0, "self": 0, "timeout": 0}
            self._obs = self.env.reset()
        if was_running:
            self.start()

    def save(self, path: Path | None = None) -> Path:
        target = path or MODEL_PATH
        with self.lock:
            self.agent.save(target)
        return target

    def load(self, path: Path | None = None) -> Path:
        target = path or MODEL_PATH
        with self.lock:
            self.agent.load(target)
        return target

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="snake-train", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        self._emit()

    def _loop(self) -> None:
        torch.set_num_threads(1)
        last_push = 0.0
        while not self._stop.is_set():
            result = self.step()
            now = time.monotonic()
            if result["done"] or now - last_push >= self._push_interval:
                self._emit()
                last_push = now

    def _emit(self) -> None:
        if self.on_update:
            self.on_update(self.status())


class Watcher:
    """用已训练 DQN 慢速贪心演示。不调用 learn。棋盘尺寸必须与训练相同。"""

    def __init__(
        self,
        agent: DqnAgent,
        lock: threading.Lock,
        grid_size: int = 8,
        hz: float = 8.0,
        on_update: UpdateFn | None = None,
    ) -> None:
        self.agent = agent
        self.lock = lock
        self.grid_size = grid_size
        self.dt = 1.0 / hz
        self.on_update = on_update
        self.env = SnakeEnv(grid_size=grid_size)
        self._obs = self.env.reset()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._obs = self.env.reset()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="snake-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        self._emit()

    def status(self) -> dict:
        return {
            "channel": "watch",
            "running": self.running,
            "snapshot": self.env.snapshot(),
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self.lock:
                if len(self.env.snake) >= HEURISTIC_MIN_LENGTH:
                    action = self.env.heuristic_action()
                else:
                    action = self.agent.act(self._obs, 0.0, self.env.legal_actions())
            result = self.env.step(action)
            self._obs = result.observation
            if result.done:
                self._obs = self.env.reset()
            self._emit()
            self._stop.wait(self.dt)

    def _emit(self) -> None:
        if self.on_update:
            self.on_update(self.status())
