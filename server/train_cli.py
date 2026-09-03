"""无界面训练自检：4 通道 DQN + 摸尾掩码 + 跟尾后，均分应明显高于约 21。

用法（在项目根目录 snake-rl/）::

    python -m server.train_cli

固定 seed=1、8×8。后 50 局均分需要 >= 28，否则退出码 1。
"""

from __future__ import annotations

import sys

import torch

from server.trainer import Trainer


def main() -> int:
    torch.set_num_threads(4)
    episodes = 6000
    trainer = Trainer(grid_size=8, seed=1)
    scores: list[int] = []
    while len(scores) < episodes:
        result = trainer.step()
        if result["done"]:
            scores.append(result["score"])
            if len(scores) % 500 == 0:
                print(
                    f"episode={len(scores)} mean50={trainer.mean_score():.2f} "
                    f"eps={trainer.epsilon:.3f} replay={trainer.agent.replay_size} "
                    f"deaths={trainer.deaths}",
                    flush=True,
                )
    first = sum(scores[:50]) / 50
    last = sum(scores[-50:]) / 50
    print(f"episodes={episodes}")
    print(f"first50_mean={first:.3f}")
    print(f"last50_mean={last:.3f}")
    print(f"replay={trainer.agent.replay_size}")
    print(f"deaths={trainer.deaths}")
    wall = trainer.deaths["wall"]
    self_hit = trainer.deaths["self"]
    if wall > self_hit:
        print("FAIL: wall deaths should drop below self-collisions with masking")
        return 1
    if last < 28:
        print("FAIL: last50 mean need >= 28")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
