"""贪吃蛇强化学习后端包。

本包是整个项目的「真相源」：游戏规则、奖励、DQN、训练循环都在这里，
Vue 前端只负责画棋盘和发 HTTP/WebSocket 请求。

文件分工
--------
- env.py       Gym 风格环境：reset / step / 网格观测
- rewards.py   奖励函数（和碰撞逻辑分开）
- agent.py     小 DQN（CNN + 回放 + 目标网络）
- trainer.py   后台线程训练 / 贪心观看
- schemas.py   FastAPI 请求与响应模型
- app.py       HTTP + WebSocket API
- train_cli.py 无界面跑若干局，检查均分是否超过表格 Q 天花板
"""
