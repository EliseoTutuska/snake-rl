---
name: snake-rl
description: >-
  snake-rl 项目约定：Python 环境与 DQN 是真相源，Vue 只做人玩/看板/观看；
  8×8 训练观看、10×10 人玩；相对动作 0/1/2；两层掩码与 BFS 启发式；不擅自上 PPO/GAE、
  不重置大脑、不改卷积通道。用户提到训练、均分、死因、掩码、启发式、看板、
  dqn.pt、iterative-study、改奖励或继续训时必须使用本技能。
---

# snake-rl

在本仓库写代码或解读训练数字时遵循下列约定。细节数字以代码和 `docs/iterative-study.md` 为准。

## 分工（不要反转）

- **Python**（`server/`）是规则、奖励、观测、DQN、训练循环的真相源。训练在进程内 `env.step()`，**不走 HTTP**。
- **Vue**（`src/`）只做人玩、看板、观看。通过 HTTP / WebSocket 读写状态，不实现碰撞或学习。
- 人玩默认 **10×10**；训练和观看必须 **8×8**（CNN 输入尺寸绑死）。不要把人玩棋盘改成训练尺寸，也不要把训练改成 10。

## 算法（默认不要换）

- 动作：相对朝向 `{0 直行, 1 右转, 2 左转}`。人玩方向键经 `abs_to_rel`。
- 观测：`observe()` 为朝上旋转的 **4×8×8**（头、身体衰减、食物、可达）。改 `OBS_CHANNELS` 会使 `server/models/dqn.pt` 作废。
- 学习：**Double DQN + PER**。不用 PPO、不用 GAE、不上 GPU，除非用户明确要求。
- 长度 **≥ 12**：训练 ε 强制 0。长度 **≥ 20**：`heuristic_action` 接管（转移仍写入回放）。
- 超时：`4n²` 步不吃（8×8 为 256）。满分 61。

## 硬约束（改过会回退）

1. `legal_actions()` 两层：先「一步不致死 ∧ 能摸尾 ∧ 洪水 ≥ 新长度」；若为空，只保留一步不致死。**禁止**全 `True` 放开墙。三者皆死则全 `False`。
2. `DqnAgent.act` 合法集为空时返回 0，不要扩成 `{0,1,2}`。
3. 启发式：空格 BFS 到食物且模拟后仍能摸尾 → 吃；否则 BFS 跟尾；再否则最大可达空间。不要改回曼哈顿贴食物。
4. **不要擅自「重置大脑」**、删 `dqn.pt`、或从零重训。权重已进 git。换通道/棋盘/卷积结构前先说明旧权作废并征求同意。
5. 改奖励只动 `server/rewards.py`。当前：eat +10，wall/self −10，timeout −4，分数 ≥10 后 idle −0.01，closer ±0.05 仅 score < 10。

## 改代码时看哪

| 要动的东西 | 文件 |
|---|---|
| 碰撞、掩码、启发式、观测 | `server/env.py` |
| 塑形 | `server/rewards.py` |
| CNN / PER / 存加载 | `server/agent.py` |
| ε、启发式接管、观看 | `server/trainer.py` |
| HTTP / WS | `server/app.py` |
| 看板 | `src/` |

改掩码或启发式后跑：`.venv/bin/python -m server.smoke_test`。门槛自检：`.venv/bin/python -m server.train_cli`（`grid_size=8`, `seed=1`）。看板：`npm run api` 与 `npm run dev`。

## 解读看板 / 更新文档

用户贴训练面板数字时：

1. 先看 **死因结构**，再看均分。墙死多 → 掩码兜底或非法动作；自撞多 → 长蛇规划；超时主导 → 会活但吃不满（当前瓶颈）。
2. 均分指近 50 局。ε 未到底时，长度 20 后分数主要来自启发式，继续降 ε 往往上不去一个大台阶。
3. 若用户要更新文档，把快照写入 `docs/iterative-study.md`（摘要、§4.5、表 1、讨论/结论），不要改计划文件。

当前已知平台（约 793 局）：均分约 **36**，超时约 94%。下一刀是超时上限或跟尾时的安全取食，不是更多 SGD，也不是换 PPO。

## 仓库

私有 GitHub：`EliseoTutuska/snake-rl`。`dqn.pt` 与 `qtable.json` **要提交**。不要把 `.venv/`、`node_modules`、`.cursor/` 里除本技能外的配置推进去。
