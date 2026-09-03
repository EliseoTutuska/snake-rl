"""FastAPI 用的 Pydantic 模型：校验请求、约束字段范围、生成 OpenAPI。

命名用 camelCase（gridSize、lastReward），与 Vue 的 TypeScript 类型、
SnakeEnv.snapshot() 的 dict 键一致，避免前端再做一层转换。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PointOut(BaseModel):
    """棋盘上的一个格子。"""

    x: int
    y: int


class SnapshotOut(BaseModel):
    """一帧完整局面，SnakeBoard.vue 只依赖这份数据。"""

    gridSize: int
    snake: list[PointOut]
    food: PointOut
    dir: int  # 0上 1右 2下 3左，与 env.ABS_* 一致
    score: int
    done: bool
    steps: int
    stepsWithoutFood: int
    lastReward: float
    lastReason: str | None = None


class PlayStepIn(BaseModel):
    """人玩一步。两种写法二选一，都出现时优先 dir。

    - action: 相对动作 0/1/2（直行/右转/左转），给调试或以后接别的客户端。
    - dir: 绝对方向 0–3，对应方向键；服务端用 abs_to_rel 转换。
    """

    action: int | None = Field(default=None, ge=0, le=2)
    dir: int | None = Field(default=None, ge=0, le=3)


class PlayStateOut(BaseModel):
    """人玩接口的统一返回：观测给算法，snapshot 给画面。"""

    observation: list[float]
    snapshot: SnapshotOut
    reward: float = 0.0
    done: bool = False
    info: dict = Field(default_factory=dict)
