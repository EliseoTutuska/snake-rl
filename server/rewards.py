"""奖励函数：把「这一步发生了什么」映射成一个标量。

事件种类（由 env.SnakeEnv 产生，再交给 compute_reward）
------------------------------------------------------
- wall / self           : 撞死，death
- timeout               : 超时，比撞死轻
- eat                   : 吃到食物；仍可叠加死胡同惩罚
- win                   : 占满整张地图
- move                  : 普通走一步；低分时用 closer，中盘起 idle，任何存活步可用 deadend

不是 RLHF 奖励模型，只是手写规则。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """一组可调的奖励系数。Trainer / SnakeEnv 共用同一份配置。

    Attributes:
        eat: 吃到食物。
        death: 撞墙、撞自己。
        timeout: 连续多步没吃到。比 death 轻，避免为了不死而空转被等同于撞死。
        step_penalty: 每步固定消耗，默认 0。
        idle_penalty: 分数达到 idle_after_score 后，普通移动再扣一点。
        idle_after_score: 超过该分数才施加 idle_penalty。
        closer: 靠近/远离食物。只在 score < closer_until_score 时生效。
        closer_until_score: 超过该分数不再给 closer，避免长蛇贴食物贴死。
        deadend: 连通空格少于蛇长时的惩罚系数（负数）。
        win_bonus: 填满棋盘时在 eat 之上再加。
    """

    eat: float = 10.0
    death: float = -10.0
    timeout: float = -4.0
    step_penalty: float = 0.0
    idle_penalty: float = -0.01
    idle_after_score: int = 10
    closer: float = 0.05
    closer_until_score: int = 10
    deadend: float = -0.6
    win_bonus: float = 50.0


def _deadend_penalty(reachable: int | None, length: int | None, cfg: RewardConfig) -> float:
    if reachable is None or length is None or length <= 0:
        return 0.0
    # 空间小于 2 倍蛇长就开始给负信号，不必等到已经钻不出来。
    limit = length * 2
    if reachable >= limit:
        return 0.0
    return cfg.deadend * (1.0 - reachable / limit)


def compute_reward(
    event: str,
    *,
    closer: bool | None = None,
    reachable: int | None = None,
    length: int | None = None,
    score: int | None = None,
    cfg: RewardConfig | None = None,
) -> float:
    """根据事件名返回标量奖励。存活步可叠加死胡同惩罚。"""
    cfg = cfg or RewardConfig()
    if event in ("wall", "self"):
        return cfg.death
    if event == "timeout":
        return cfg.timeout
    if event == "win":
        return cfg.eat + cfg.win_bonus

    reward = cfg.eat if event == "eat" else cfg.step_penalty
    if event == "move" and score is not None and score >= cfg.idle_after_score:
        reward += cfg.idle_penalty
    use_closer = score is None or score < cfg.closer_until_score
    if event == "move" and use_closer:
        if closer is True:
            reward += cfg.closer
        elif closer is False:
            reward -= cfg.closer
    reward += _deadend_penalty(reachable, length, cfg)
    return reward
