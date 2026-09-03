"""HTTP / WebSocket 入口：把「人玩、训练、观看」接到 Python 环境上。

进程内单例
----------
- play_env  人玩的 10×10 环境，由浏览器定时 POST /play/step 驱动。
- trainer   8×8 DQN 训练循环，step 在内存里跑。
- watcher   共用 trainer.agent，ε=0，同样 8×8。
- hub       训练线程 → asyncio 队列 → 每个 WebSocket 客户端。

训练不要逐步打 HTTP：那会把 5090/CPU 耗在网络上。本文件里 /train/start
只负责「打开线程」，真正的 env.step 在 trainer._loop。

开发时 Vite 把 /api 代理到本服务 :8000。若存在 ../dist，则把打包后的 Vue
挂到 /，一条 uvicorn 就能同时提供页面和 API。
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.env import SnakeEnv, abs_to_rel
from server.schemas import PlayStateOut, PlayStepIn
from server.trainer import MODEL_PATH, Trainer, Watcher


class Hub:
    """线程安全的广播器：训练线程调用 broadcast，事件循环里把消息放入各连接的 Queue。"""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self._queues: list[asyncio.Queue] = []
        self._lock = threading.Lock()

    def add(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._queues.append(queue)

    def remove(self, queue: asyncio.Queue) -> None:
        with self._lock:
            if queue in self._queues:
                self._queues.remove(queue)

    def broadcast(self, message: dict) -> None:
        """可从任意线程调用。事件循环尚未就绪（启动前）时直接丢弃。"""
        loop = self.loop
        if loop is None:
            return
        with self._lock:
            queues = list(self._queues)
        for queue in queues:
            loop.call_soon_threadsafe(self._put, queue, message)

    @staticmethod
    def _put(queue: asyncio.Queue, message: dict) -> None:
        """队列满则丢掉最旧的一条再写入，保证前端看到的是较新状态而不是堵住训练。"""
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass


hub = Hub()
# 训练写权重、观看读权重，必须同一把锁。
agent_lock = threading.Lock()
trainer = Trainer(grid_size=8, lock=agent_lock, on_update=hub.broadcast)
play_env = SnakeEnv(grid_size=10)
watcher = Watcher(
    agent=trainer.agent,
    lock=agent_lock,
    grid_size=8,
    on_update=hub.broadcast,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时记下事件循环供 Hub 用；关闭时停掉训练/观看线程，避免热重载残留。"""
    hub.loop = asyncio.get_running_loop()
    yield
    watcher.stop()
    trainer.stop()


app = FastAPI(title="Snake RL", lifespan=lifespan)
# 开发时即使不走 Vite 代理、直连 :8000 也能被 Vue 调用。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _play_payload(reward: float = 0.0, done: bool | None = None) -> dict:
    """组装人玩接口的 JSON。reward/done 可覆盖（例如刚 step 完要用这一步的结果）。"""
    snap = play_env.snapshot()
    return {
        "observation": play_env.observe(),
        "snapshot": snap,
        "reward": reward,
        "done": play_env.done if done is None else done,
        "info": {
            "score": play_env.score,
            "length": len(play_env.snake),
            "reason": play_env.last_reason,
        },
    }


@app.post("/api/play/reset", response_model=PlayStateOut)
def play_reset() -> dict:
    """新开一局人玩。必须用 POST；浏览器 GET 会 405。"""
    play_env.reset()
    payload = _play_payload()
    hub.broadcast({"channel": "play", **payload})
    return payload


@app.get("/api/play/state", response_model=PlayStateOut)
def play_state() -> dict:
    """只读当前人玩局面，供页面刷新后恢复棋盘。"""
    return _play_payload()


@app.post("/api/play/step", response_model=PlayStateOut)
def play_step(body: PlayStepIn) -> dict:
    """人玩一步。优先把绝对方向转成相对动作，否则用 body.action，都缺则直行。"""
    if body.dir is not None:
        action = abs_to_rel(play_env.dir, body.dir)
    elif body.action is not None:
        action = body.action
    else:
        action = 0
    result = play_env.step(action)
    payload = _play_payload(reward=result.reward, done=result.done)
    hub.broadcast({"channel": "play", **payload})
    return payload


@app.post("/api/train/start")
def train_start() -> dict:
    """打开内存训练循环。观看会先停掉，避免和训练抢同一张 Q 表。"""
    watcher.stop()
    trainer.start()
    return trainer.status()


@app.post("/api/train/stop")
def train_stop() -> dict:
    trainer.stop()
    return trainer.status()


@app.post("/api/train/reset")
def train_reset() -> dict:
    """清空网络、回放池和局数（前端「重置大脑」）。"""
    trainer.reset_learning()
    return trainer.status()


@app.get("/api/train/status")
def train_status() -> dict:
    return trainer.status()


@app.post("/api/train/save")
def train_save() -> dict:
    """把当前 DQN 权重写到 server/models/dqn.pt。"""
    path = trainer.save()
    return {"ok": True, "path": str(path), "qSize": trainer.agent.replay_size}


@app.post("/api/train/load")
def train_load() -> dict:
    """从默认路径加载 DQN 权重。文件不存在时 404。"""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="还没有保存过 DQN 权重")
    try:
        trainer.load()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return trainer.status()


@app.post("/api/watch/start")
def watch_start() -> dict:
    """贪心回放。会先停训练，避免一边更新 Q 一边演示。"""
    trainer.stop()
    watcher.start()
    return watcher.status()


@app.post("/api/watch/stop")
def watch_stop() -> dict:
    watcher.stop()
    return watcher.status()


@app.get("/api/watch/status")
def watch_status() -> dict:
    return watcher.status()


@app.websocket("/api/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """每个浏览器一条约连接。先推 play/train/watch 各一份，再阻塞读广播队列。"""
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=32)
    hub.add(queue)
    try:
        await ws.send_json({"channel": "play", **_play_payload()})
        await ws.send_json(trainer.status())
        await ws.send_json(watcher.status())
        while True:
            message = await queue.get()
            await ws.send_json(message)
    except WebSocketDisconnect:
        pass
    finally:
        hub.remove(queue)


# 生产构建后：API 路由注册完再挂静态资源，避免 /* 盖住 /api。
dist_dir = Path(__file__).resolve().parent.parent / "dist"
if dist_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="spa")
