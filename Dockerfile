# 可替换的基础镜像（国内拉取慢时可换成镜像站前缀）
ARG NODE_IMAGE=node:24-bookworm-slim
ARG PYTHON_IMAGE=python:3.12-slim

# ---------- 阶段 1：构建 Vue 前端 ----------
FROM ${NODE_IMAGE} AS web
WORKDIR /app

ARG NPM_REGISTRY=https://registry.npmjs.org
COPY package.json package-lock.json ./
RUN npm config set registry "$NPM_REGISTRY" && npm ci

COPY index.html env.d.ts tsconfig.json tsconfig.app.json tsconfig.node.json vite.config.ts ./
COPY public ./public
COPY src ./src
RUN npm run build

# ---------- 阶段 2：Python 运行时（API + WebSocket + 静态页） ----------
FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
# 先从 CPU 源装 torch，避免 PyPI 的 Linux 版把整套 CUDA 依赖拖进来（镜像会大好几个 GB）。
RUN pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" torch \
 && pip install --no-cache-dir --index-url "$PIP_INDEX_URL" -r requirements.txt

COPY server ./server
COPY --from=web /app/dist ./dist

EXPOSE 8000

# 训练状态在进程内存里，必须单 worker，不能加 --workers。
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
