"""短自检：4 通道观测、摸尾掩码、启发式、拒绝旧权重。不跑满训练。

用法（项目根目录）::

    python -m server.smoke_test
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from server.agent import DqnAgent
from server.env import (
    ABS_LEFT,
    HEURISTIC_MIN_LENGTH,
    OBS_CHANNELS,
    Point,
    SnakeEnv,
)
from server.rewards import compute_reward


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def _check_obs_and_wall() -> str | None:
    env = SnakeEnv(grid_size=8, seed=0)
    obs = env.observe()
    expect = OBS_CHANNELS * 8 * 8
    if len(obs) != expect:
        return f"观测长度 {len(obs)}，期望 {expect}"
    env.snake = [Point(0, 4), Point(1, 4), Point(2, 4)]
    env.dir = ABS_LEFT
    env.food = Point(5, 5)
    env.done = False
    mask = env.legal_actions()
    if mask[0] is not False:
        return f"贴左墙朝左时直行应非法，mask={mask}"
    if not (mask[1] or mask[2]):
        return f"贴左墙朝左时应能转弯，mask={mask}"
    # 三面都一步就死：不得把直行改成合法。
    env.snake = [Point(0, 0), Point(0, 1), Point(0, 2)]
    env.dir = ABS_LEFT
    env.food = Point(4, 4)
    trapped = env.legal_actions()
    if trapped[0] is not False:
        return f"角落全非法时直行不得放开，mask={trapped}"
    if any(trapped):
        return f"角落三面皆死应全 False，mask={trapped}"
    return None


def _check_pocket_blocks_eat() -> str | None:
    """体积够但尾巴困在环心：摸尾检查应判非法。"""
    env = SnakeEnv(grid_size=8, seed=0)
    body = [
        Point(0, 0),
        Point(1, 0),
        Point(2, 0),
        Point(3, 0),
        Point(3, 1),
        Point(3, 2),
        Point(3, 3),
        Point(4, 3),
        Point(5, 3),
        Point(5, 4),
        Point(5, 5),
        Point(4, 5),
        Point(3, 5),
        Point(3, 4),
        Point(4, 4),
    ]
    if env._can_reach_tail(body):
        return "环心尾巴仍判定能摸到"
    if env._reachable_of(body) < len(body):
        return "环外体积应仍够蛇身，以便区分摸尾检查"
    # 吃进环心：头在 (3,4) 朝右，食物在 (4,4)。
    env.snake = list(reversed(body[:-1]))
    env.dir = 1
    env.food = Point(4, 4)
    env.done = False
    env.score = 12
    simulated = env._simulate(0)
    if simulated is None:
        return "吃进环心应能模拟，不应一步撞死"
    after, eating = simulated
    if not eating:
        return "环心用例应是吃豆"
    if env._can_reach_tail(after):
        return "吃进环心后仍判定能摸到尾巴"
    mask = env.legal_actions()
    if mask == [True, True, True]:
        return f"摸不到尾巴时不应放开全部动作，mask={mask}"
    if mask[0] is not True:
        return f"吃进环心一步能活，应保留直行，mask={mask}"
    if mask[1] or mask[2]:
        return f"转弯撞身应仍非法，mask={mask}"
    return None


def _check_heuristic() -> str | None:
    env = SnakeEnv(grid_size=8, seed=1)
    cells: list[Point] = []
    for y in range(8):
        xs = range(8) if y % 2 == 0 else range(7, -1, -1)
        for x in xs:
            cells.append(Point(x, y))
    env.snake = list(reversed(cells[:HEURISTIC_MIN_LENGTH]))
    env.dir = 1
    env.food = Point(7, 7)
    env.done = False
    if len(env.snake) < HEURISTIC_MIN_LENGTH:
        return f"启发式用例蛇长 {len(env.snake)} < {HEURISTIC_MIN_LENGTH}"
    mask = env.legal_actions()
    action = env.heuristic_action(mask)
    if env._simulate(action) is None:
        return f"heuristic_action={action} 一步就会死"
    if mask[action] is False and any(mask):
        return f"heuristic_action={action} 不在合法集合 {mask}"
    return None


def _check_rewards() -> str | None:
    if compute_reward("timeout") != -4.0:
        return f"timeout 应为 -4，得到 {compute_reward('timeout')}"
    if compute_reward("self") != -10.0:
        return f"self 应为 -10，得到 {compute_reward('self')}"
    move = compute_reward("move", closer=True, score=12, reachable=40, length=13)
    if move > -0.005:
        return f"score>=10 的 move 应含 idle 惩罚，得到 {move}"
    return None


def _check_old_weights() -> str | None:
    agent = DqnAgent(grid_size=8)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old.pt"
        import torch

        torch.save(
            {
                "grid_size": 8,
                "gamma": 0.97,
                "online": agent.online.state_dict(),
                "target": agent.target.state_dict(),
            },
            path,
        )
        # 去掉 channels 字段，模拟旧 3 通道存档；state_dict 仍是 4 通道，
        # load 应在读到缺 channels 时先拒绝。
        try:
            agent.load(path)
        except ValueError as exc:
            if "3 通道" not in str(exc):
                return f"旧权重错误信息不对: {exc}"
        else:
            return "旧权重缺少 channels 时应当拒绝加载"

    from server.trainer import MODEL_PATH

    if MODEL_PATH.exists():
        try:
            DqnAgent(grid_size=8).load(MODEL_PATH)
        except ValueError as exc:
            if "3 通道" not in str(exc) and "棋盘" not in str(exc):
                return f"磁盘 dqn.pt 加载错误: {exc}"
    return None


def _check_learn_and_new_save() -> str | None:
    env = SnakeEnv(grid_size=8, seed=0)
    agent = DqnAgent(grid_size=8, warmup=8, batch_size=8)
    obs = env.reset()
    for _ in range(40):
        mask = env.legal_actions()
        action = agent.act(obs, 1.0, mask)
        result = env.step(action)
        next_mask = [True, True, True] if result.done else env.legal_actions()
        agent.learn(
            obs,
            action,
            result.reward,
            result.observation,
            result.done,
            mask,
            next_mask,
        )
        obs = env.reset() if result.done else result.observation
    if agent.replay_size < 8:
        return "回放池应已写入"
    if agent.learn_steps < 8:
        return "应已开始 optimize"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "new.pt"
        agent.save(path)
        other = DqnAgent(grid_size=8)
        try:
            other.load(path)
        except ValueError as exc:
            return f"新 4 通道权重应能加载: {exc}"
    return None


def main() -> int:
    checks = (
        _check_obs_and_wall,
        _check_pocket_blocks_eat,
        _check_heuristic,
        _check_rewards,
        _check_old_weights,
        _check_learn_and_new_save,
    )
    for fn in checks:
        err = fn()
        if err:
            return _fail(err)
        print(f"ok {fn.__name__}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
