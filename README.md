# 贪吃蛇 RL

8×8 贪吃蛇上的迭代式强化学习实验：Python 是环境和训练的真相源，Vue 做人玩、看板和观看。动作始终是三个相对动作（直行 / 右转 / 左转），算法是 **朝上网格观测 + 小 DQN**（Mac CPU），不用 PPO / GAE。

当前配方：4 通道卷积、Double DQN、优先回放、两层动作掩码、长度 ≥ 20 时安全 BFS 吃豆否则跟尾。看板约 793 局近 50 局均分约 **36**，死因以超时为主；8×8 满分 61。过程见 [docs/iterative-study.md](docs/iterative-study.md)。

**在线看板**：<https://snake.mingstep.top>

## 分工

- **Python**：碰撞、食物、奖励、CNN 学 Q。训练循环在进程内 `env.step()`，不走 HTTP。
- **Vue**：方向键、画棋盘、开始/暂停训练、WebSocket 收快照。

人玩 10×10；训练和观看必须同为 8×8（卷积输入尺寸固定）。

## 启动

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install

# 终端 1：环境 + 训练 API
npm run api

# 终端 2：看板
npm run dev
```

浏览器打开 `http://localhost:5173`。

无界面自检（`seed=1`，约 6000 局）：

```sh
.venv/bin/python -m server.smoke_test
.venv/bin/python -m server.train_cli
```

仓库里带一份训练好的 `server/models/dqn.pt`（约 793 局、近 50 局均分约 36），clone 后可直接在看板里观看。继续训练会覆盖该文件。早期表格 Q 的 `server/models/qtable.json` 也一并保留。

## 文档

| 文件 | 内容 |
|---|---|
| [docs/iterative-study.md](docs/iterative-study.md) | 从表格 Q 到掩码 DQN 的迭代实验报告 |
| [docs/workflow.md](docs/workflow.md) | 强化学习工作流 |
| [docs/games-for-rl.md](docs/games-for-rl.md) | 后续可练的游戏 |
| [docs/algorithms-for-games.md](docs/algorithms-for-games.md) | DQN / PPO / AlphaZero 等与游戏的对应 |

## 奖励（`server/rewards.py`）

吃到 +10，撞墙/自撞 −10，超时 −4，分数 ≥ 10 后每步 −0.01。短蛇阶段靠近食物 ±0.05。改塑形只动这个文件。
