"""贪吃蛇环境（Gym 风格）：reset / step / observe。

和 Vue 的关系
------------
浏览器不跑这份规则。人玩时 Vue 每隔约 120ms POST /api/play/step，
由本模块在 Python 进程里结算。训练时 Trainer 直接调用 step()，不走 HTTP。

动作空间（相对当前朝向，共 3 个）
--------------------------------
0 = 直行，1 = 右转，2 = 左转。

相对动作的好处：智能体不会「倒车咬到自己」；人玩时的方向键会先经
abs_to_rel 转成这 3 个动作（按反方向会被当成直行）。

观测（给 DQN，不是截图像素）
--------------------------
observe() 返回长度 ``4 * H * W`` 的 float 列表，通道顺序：头、身体、食物、可达。

网格会旋到「当前朝向 = 画面上方」，这样相对动作（直行/左转/右转）
与卷积看到的方向一致。训练和观看必须同一 grid_size。

11 维开关仍可由 observe_features() 得到，训练主路径不用。
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from server.rewards import RewardConfig, compute_reward

# 相对动作：智能体（和人玩转换后）真正传给 step() 的值。
REL_STRAIGHT = 0
REL_RIGHT = 1
REL_LEFT = 2

# 绝对朝向，与 Vue 棋盘 dir-0..3 以及 DIR_VEC 下标一致。
ABS_UP = 0
ABS_RIGHT = 1
ABS_DOWN = 2
ABS_LEFT = 3

# 索引 = 绝对朝向；值 = (dx, dy)。y 向下增大，和屏幕坐标一致。
DIR_VEC = ((0, -1), (1, 0), (0, 1), (-1, 0))

# 局终原因：eat / wall / self / timeout / win；普通移动为 None。
Reason = str

# 与 DqnAgent 输入通道数一致；改这个必须重新训练。
OBS_CHANNELS = 4
# 达到该长度后 Trainer / Watcher 改用 heuristic_action（跟尾 / 安全吃豆）。
HEURISTIC_MIN_LENGTH = 20


@dataclass
class Point:
    """格子坐标。原点在左上，x 向右，y 向下。"""

    x: int
    y: int


@dataclass
class StepInfo:
    """step() 附带的诊断信息，训练可以只用 score，Vue 用来显示。"""

    score: int
    length: int
    reason: Reason | None = None


@dataclass
class StepResult:
    """一次 step 的完整返回，对应 Gym 的 obs, reward, terminated, info。"""

    observation: list[float]
    reward: float
    done: bool
    info: StepInfo


def turn(direction: int, action: int) -> int:
    """把「当前朝向 + 相对动作」变成新的绝对朝向。

    右转：(dir + 1) % 4，例如上(0)→右(1)→下(2)→左(3)→上。
    左转：(dir + 3) % 4，等价于 -1。
    """
    if action == REL_STRAIGHT:
        return direction
    if action == REL_RIGHT:
        return (direction + 1) % 4
    return (direction + 3) % 4


def abs_to_rel(current: int, wanted: int) -> int:
    """人玩：方向键是绝对方向，环境只要相对动作。

    若玩家按了正后方（倒车），返回直行，避免一步掉头撞身体。
    """
    if wanted == current:
        return REL_STRAIGHT
    if wanted == (current + 1) % 4:
        return REL_RIGHT
    if wanted == (current + 3) % 4:
        return REL_LEFT
    return REL_STRAIGHT


def heading_up_coords(x: int, y: int, direction: int, grid_size: int) -> tuple[int, int]:
    """把世界坐标旋到「当前朝向朝上」。用于网格观测。"""
    n = grid_size
    if direction == ABS_UP:
        return x, y
    if direction == ABS_RIGHT:
        return y, n - 1 - x
    if direction == ABS_DOWN:
        return n - 1 - x, n - 1 - y
    return n - 1 - y, x


def manhattan(a: Point, b: Point) -> int:
    """曼哈顿距离 |dx|+|dy|。用于「这一步有没有更靠近食物」。"""
    return abs(a.x - b.x) + abs(a.y - b.y)


class SnakeEnv:
    """一份独立的贪吃蛇对局。人玩、训练、观看应各用一个实例，互不踩状态。"""

    def __init__(
        self,
        grid_size: int = 10,
        seed: int | None = None,
        rewards: RewardConfig | None = None,
        max_steps_without_food: int | None = None,
    ) -> None:
        """Args:
            grid_size: 边长。人玩默认 10；训练和观看必须同尺寸（CNN 输入固定），默认 8。
            seed: RNG 种子。None 则不可复现；CLI 验学习曲线时会固定种子。
            rewards: 奖励系数。None 用默认塑形。
            max_steps_without_food: 连续这么多步没吃到就 timeout。
                默认 4 * grid_size²，给长蛇绕路。
        """
        self.grid_size = grid_size
        self.rewards = rewards or RewardConfig()
        self.max_steps_without_food = (
            max_steps_without_food
            if max_steps_without_food is not None
            else grid_size * grid_size * 4
        )
        self.rng = random.Random(seed)
        self.snake: list[Point] = []
        self.food = Point(0, 0)
        self.dir = ABS_RIGHT
        self.score = 0
        self.done = False
        self.steps = 0
        self.steps_without_food = 0
        self.last_reward = 0.0
        self.last_reason: Reason | None = None
        self.reset()

    def reset(self) -> list[float]:
        """开新局：蛇放在中心朝右，长度 3，随机放一个食物。

        Returns:
            初始网格观测，可直接交给 DqnAgent.act。
        """
        mid = self.grid_size // 2
        self.dir = ABS_RIGHT
        # 头部在中心，身体向左排，朝右走第一步不会立刻撞自己。
        self.snake = [
            Point(mid, mid),
            Point(mid - 1, mid),
            Point(mid - 2, mid),
        ]
        self.score = 0
        self.done = False
        self.steps = 0
        self.steps_without_food = 0
        self.last_reward = 0.0
        self.last_reason = None
        self._place_food()
        return self.observe()

    def step(self, action: int) -> StepResult:
        """执行一个相对动作并结算。已结束的对局再 step 只返回 done，不再移动。"""
        if self.done:
            return StepResult(
                observation=self.observe(),
                reward=0.0,
                done=True,
                info=StepInfo(score=self.score, length=len(self.snake)),
            )

        self.dir = turn(self.dir, int(action))
        head = self.snake[0]
        dx, dy = DIR_VEC[self.dir]
        nxt = Point(head.x + dx, head.y + dy)
        prev_dist = manhattan(head, self.food)

        if self._out_of_bounds(nxt):
            return self._finish("wall")

        eating = nxt.x == self.food.x and nxt.y == self.food.y
        # 吃东西时尾巴不缩，身体变长，所以碰撞检测要把尾巴格子也算占用。
        if self._hits_body(nxt, growing=eating):
            return self._finish("self")

        self.snake.insert(0, nxt)
        self.steps += 1

        if eating:
            self.score += 1
            self.steps_without_food = 0
            if len(self.snake) == self.grid_size * self.grid_size:
                return self._apply("win", done=True)
            self._place_food()
            return self._apply("eat", done=False, reachable=self._reachable())

        # 没吃到：去掉尾巴，长度不变。
        self.snake.pop()
        self.steps_without_food += 1
        if self.steps_without_food >= self.max_steps_without_food:
            return self._finish("timeout")

        next_dist = manhattan(nxt, self.food)
        return self._apply(
            "move",
            done=False,
            closer=next_dist < prev_dist,
            reachable=self._reachable(),
        )

    def legal_actions(self) -> list[bool]:
        """两层掩码：先要能摸到尾巴；否则只保留一步不致死，绝不把撞墙改成合法。"""
        length = len(self.snake)
        survive: list[bool] = []
        safe: list[bool] = []
        for action in (REL_STRAIGHT, REL_RIGHT, REL_LEFT):
            simulated = self._simulate(action)
            if simulated is None:
                survive.append(False)
                safe.append(False)
                continue
            survive.append(True)
            body, eating = simulated
            need = length + (1 if eating else 0)
            safe.append(self._can_reach_tail(body) and self._reachable_of(body) >= need)
        if any(safe):
            return safe
        return survive

    def heuristic_action(self, mask: list[bool] | None = None) -> int:
        """长蛇：先走「模拟后仍能摸尾」的食物 BFS 第一步，否则跟尾，再否则挑空间最大。"""
        survive = [a for a in (REL_STRAIGHT, REL_RIGHT, REL_LEFT) if self._simulate(a) is not None]
        candidates = survive
        if mask is not None:
            masked = [a for a in survive if mask[a]]
            if masked:
                candidates = masked
        if not candidates:
            return REL_STRAIGHT

        food_cell = self._bfs_first_toward(self.food, allow_tail=False)
        if food_cell is not None:
            action = self._rel_for_cell(food_cell)
            if action in candidates:
                simulated = self._simulate(action)
                if simulated is not None and self._can_reach_tail(simulated[0]):
                    return action

        tail_cell = self._bfs_first_toward(self.snake[-1], allow_tail=True)
        if tail_cell is not None:
            action = self._rel_for_cell(tail_cell)
            if action in candidates:
                simulated = self._simulate(action)
                if simulated is not None:
                    return action

        best = candidates[0]
        best_space = -1
        for action in candidates:
            simulated = self._simulate(action)
            if simulated is None:
                continue
            space = self._reachable_of(simulated[0])
            if space > best_space:
                best_space = space
                best = action
        return best

    def _simulate(self, action: int) -> tuple[list[Point], bool] | None:
        """相对动作走一步后的身体；一步就死者返回 None。"""
        head = self.snake[0]
        new_dir = turn(self.dir, action)
        dx, dy = DIR_VEC[new_dir]
        nxt = Point(head.x + dx, head.y + dy)
        eating = nxt.x == self.food.x and nxt.y == self.food.y
        if self._out_of_bounds(nxt) or self._hits_body(nxt, growing=eating):
            return None
        if eating:
            return [nxt, *self.snake], True
        return [nxt, *self.snake[:-1]], False

    def _reachable(self) -> int:
        return self._reachable_of(self.snake)

    def _reachable_cells(self, body: list[Point]) -> set[tuple[int, int]]:
        """从头部出发的连通格（含头、食物，不含其余身体）。"""
        n = self.grid_size
        head = body[0]
        blocked = {(p.x, p.y) for p in body[1:]}
        seen = {(head.x, head.y)}
        queue: deque[tuple[int, int]] = deque([(head.x, head.y)])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIR_VEC:
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= n or ny >= n:
                    continue
                cell = (nx, ny)
                if cell in blocked or cell in seen:
                    continue
                seen.add(cell)
                queue.append(cell)
        return seen

    def _reachable_of(self, body: list[Point]) -> int:
        """从身体头部出发的连通格数（含头、食物，不含其余身体）。"""
        return len(self._reachable_cells(body))

    def _can_reach_tail(self, body: list[Point]) -> bool:
        """走完后头能否摸到尾巴（中间身体当墙，尾巴当终点）。"""
        return self._tail_distance(body) < 10**9

    def _tail_distance(self, body: list[Point]) -> int:
        """头部到尾巴的格子距离；不可达则为 10**9。"""
        if len(body) <= 1:
            return 0
        n = self.grid_size
        head = body[0]
        tail = body[-1]
        if head.x == tail.x and head.y == tail.y:
            return 0
        blocked = {(p.x, p.y) for p in body[1:-1]}
        start = (head.x, head.y)
        goal = (tail.x, tail.y)
        dist = {start: 0}
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            x, y = queue.popleft()
            d = dist[(x, y)]
            for dx, dy in DIR_VEC:
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= n or ny >= n:
                    continue
                cell = (nx, ny)
                if cell in blocked or cell in dist:
                    continue
                dist[cell] = d + 1
                if cell == goal:
                    return d + 1
                queue.append(cell)
        return 10**9

    def _bfs_first_toward(self, goal: Point, *, allow_tail: bool) -> Point | None:
        """从头部 BFS 到 goal 的第一步格子。allow_tail 时尾巴可走、中间身体当墙。"""
        n = self.grid_size
        head = self.snake[0]
        start = (head.x, head.y)
        goal_c = (goal.x, goal.y)
        if start == goal_c:
            return None
        if allow_tail:
            blocked = {(p.x, p.y) for p in self.snake[1:-1]} if len(self.snake) > 1 else set()
        else:
            blocked = {(p.x, p.y) for p in self.snake[1:]}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIR_VEC:
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= n or ny >= n:
                    continue
                cell = (nx, ny)
                if cell in blocked or cell in parent:
                    continue
                parent[cell] = (x, y)
                if cell == goal_c:
                    cur = cell
                    while parent[cur] != start:
                        nxt = parent[cur]
                        if nxt is None:
                            return None
                        cur = nxt
                    return Point(cur[0], cur[1])
                queue.append(cell)
        return None

    def _rel_for_cell(self, nxt: Point) -> int | None:
        """把相邻一格转成相对动作；正后方返回 None（不能倒车）。"""
        head = self.snake[0]
        dx, dy = nxt.x - head.x, nxt.y - head.y
        for abs_dir, vec in enumerate(DIR_VEC):
            if vec != (dx, dy):
                continue
            if abs_dir == self.dir:
                return REL_STRAIGHT
            if abs_dir == (self.dir + 1) % 4:
                return REL_RIGHT
            if abs_dir == (self.dir + 3) % 4:
                return REL_LEFT
            return None
        return None

    def observe(self) -> list[float]:
        """四通道棋盘（头 / 身 / 食物 / 可达），已旋到朝向朝上。"""
        n = self.grid_size
        cells = n * n
        head_ch = [0.0] * cells
        body_ch = [0.0] * cells
        food_ch = [0.0] * cells
        reach_ch = [0.0] * cells
        length = len(self.snake)
        for i, p in enumerate(self.snake):
            rx, ry = heading_up_coords(p.x, p.y, self.dir, n)
            idx = ry * n + rx
            if i == 0:
                head_ch[idx] = 1.0
            else:
                body_ch[idx] = (length - i) / length
        fx, fy = heading_up_coords(self.food.x, self.food.y, self.dir, n)
        food_ch[fy * n + fx] = 1.0
        for x, y in self._reachable_cells(self.snake):
            rx, ry = heading_up_coords(x, y, self.dir, n)
            reach_ch[ry * n + rx] = 1.0
        return head_ch + body_ch + food_ch + reach_ch

    def observe_features(self) -> list[int]:
        """旧版 11 维 0/1 特征，留给对照实验。训练主路径不用。"""
        head = self.snake[0]
        danger_straight = self._danger_in(self.dir)
        danger_right = self._danger_in((self.dir + 1) % 4)
        danger_left = self._danger_in((self.dir + 3) % 4)
        return [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(self.dir == ABS_LEFT),
            int(self.dir == ABS_RIGHT),
            int(self.dir == ABS_UP),
            int(self.dir == ABS_DOWN),
            int(self.food.x < head.x),
            int(self.food.x > head.x),
            int(self.food.y < head.y),
            int(self.food.y > head.y),
        ]

    def snapshot(self) -> dict:
        """给 Vue 画格子用的完整局面。字段名用 camelCase，与前端类型对齐。"""
        return {
            "gridSize": self.grid_size,
            "snake": [{"x": p.x, "y": p.y} for p in self.snake],
            "food": {"x": self.food.x, "y": self.food.y},
            "dir": self.dir,
            "score": self.score,
            "done": self.done,
            "steps": self.steps,
            "stepsWithoutFood": self.steps_without_food,
            "lastReward": self.last_reward,
            "lastReason": self.last_reason,
        }

    def _apply(
        self,
        event: str,
        *,
        done: bool,
        closer: bool | None = None,
        reachable: int | None = None,
    ) -> StepResult:
        """调用奖励函数、更新 last_* / done，再打包 StepResult。"""
        reward = compute_reward(
            event,
            closer=closer,
            reachable=reachable,
            length=len(self.snake),
            score=self.score,
            cfg=self.rewards,
        )
        self.last_reward = reward
        # 普通移动不写 last_reason，避免前端一直显示「move」。
        self.last_reason = event if event != "move" else None
        self.done = done
        return StepResult(
            observation=self.observe(),
            reward=reward,
            done=done,
            info=StepInfo(
                score=self.score,
                length=len(self.snake),
                reason=self.last_reason,
            ),
        )

    def _finish(self, reason: Reason) -> StepResult:
        """失败结局的快捷方式：给 death 奖励并结束本局。"""
        return self._apply(reason, done=True)

    def _danger_in(self, direction: int) -> bool:
        """从头部沿该绝对方向走一格是否会死。用于观测里的 danger 位。"""
        head = self.snake[0]
        dx, dy = DIR_VEC[direction]
        nxt = Point(head.x + dx, head.y + dy)
        return self._out_of_bounds(nxt) or self._hits_body(nxt, growing=False)

    def _out_of_bounds(self, p: Point) -> bool:
        return p.x < 0 or p.y < 0 or p.x >= self.grid_size or p.y >= self.grid_size

    def _hits_body(self, p: Point, *, growing: bool) -> bool:
        """下一格是否落在身体上。

        不生长时尾巴这一帧会腾出来，所以不把最后一节算作障碍。
        生长（正在吃）时尾巴留着，整条身体都占格。
        """
        limit = len(self.snake) if growing else len(self.snake) - 1
        return any(cell.x == p.x and cell.y == p.y for cell in self.snake[:limit])

    def _place_food(self) -> None:
        """在未占用的格子里均匀随机放食物。棋盘已满则随便放 (0,0)（随后会判 win）。"""
        taken = {(p.x, p.y) for p in self.snake}
        empty = [
            Point(x, y)
            for y in range(self.grid_size)
            for x in range(self.grid_size)
            if (x, y) not in taken
        ]
        if not empty:
            self.food = Point(0, 0)
            return
        self.food = self.rng.choice(empty)
