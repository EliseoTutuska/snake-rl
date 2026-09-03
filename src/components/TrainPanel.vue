<script setup lang="ts">
import SnakeBoard from './SnakeBoard.vue'
import ScoreChart from './ScoreChart.vue'
import * as api from '../api'
import type { TrainStatus } from '../types'

defineProps<{
  status: TrainStatus
}>()

const emit = defineEmits<{
  'update:status': [TrainStatus]
  error: [string]
  info: [string]
}>()

async function wrap<T>(fn: () => Promise<T>, after?: (value: T) => void) {
  try {
    const value = await fn()
    after?.(value)
  } catch (err) {
    emit('error', err instanceof Error ? err.message : String(err))
  }
}

function start() {
  wrap(api.trainStart, (status) => emit('update:status', status))
}

function stop() {
  wrap(api.trainStop, (status) => emit('update:status', status))
}

function reset() {
  wrap(api.trainReset, (status) => emit('update:status', status))
}

function save() {
  wrap(api.trainSave, (res) => emit('info', `已保存权重（回放池 ${res.qSize}）→ ${res.path}`))
}

function load() {
  wrap(api.trainLoad, (status) => {
    emit('update:status', status)
    emit('info', `已加载 DQN 权重，回放池 ${status.qSize}`)
  })
}
</script>

<template>
  <section class="panel">
    <SnakeBoard :snapshot="status.snapshot" compact />
    <aside>
      <dl>
        <div>
          <dt>局数</dt>
          <dd>{{ status.episode }}</dd>
        </div>
        <div>
          <dt>近 50 局均分</dt>
          <dd>{{ status.meanScore.toFixed(2) }}</dd>
        </div>
        <div>
          <dt>ε</dt>
          <dd>{{ status.epsilon.toFixed(3) }}</dd>
        </div>
        <div>
          <dt>回放池</dt>
          <dd>{{ status.qSize }}</dd>
        </div>
        <div>
          <dt>总步数</dt>
          <dd>{{ status.totalSteps }}</dd>
        </div>
        <div>
          <dt>状态</dt>
          <dd class="small">{{ status.running ? '训练中' : '已停止' }}</dd>
        </div>
      </dl>
      <ScoreChart :scores="status.scores" />
      <p v-if="status.deaths" class="hint">
        死因 撞墙 {{ status.deaths.wall }} / 撞自己 {{ status.deaths.self }} / 超时
        {{ status.deaths.timeout }}
        <span v-if="status.device"> · {{ status.device }}</span>
      </p>
      <p class="hint">
        网格 DQN 在 Python 进程内 <code>env.step()</code>，不走 HTTP。页面只通过 WebSocket 收快照和曲线。
      </p>
      <div class="row">
        <button class="primary" type="button" :disabled="status.running" @click="start">开始训练</button>
        <button type="button" :disabled="!status.running" @click="stop">暂停</button>
        <button class="danger" type="button" @click="reset">重置大脑</button>
      </div>
      <div class="row">
        <button type="button" @click="save">保存权重</button>
        <button type="button" @click="load">加载权重</button>
      </div>
    </aside>
  </section>
</template>

<style scoped>
.panel {
  display: flex;
  flex-wrap: wrap;
  gap: 28px;
  align-items: flex-start;
}

aside {
  flex: 1;
  min-width: 260px;
}

dl {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 0 0 16px;
}

dt {
  color: #8b97a8;
  font-size: 12px;
}

dd {
  margin: 4px 0 0;
  font-size: 24px;
  font-variant-numeric: tabular-nums;
}

dd.small {
  font-size: 16px;
  padding-top: 6px;
}

.hint {
  color: #8b97a8;
  font-size: 13px;
  line-height: 1.55;
}

.row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

code {
  font-size: 12px;
  color: #fbbf24;
}
</style>
