<script setup lang="ts">
import SnakeBoard from './SnakeBoard.vue'
import * as api from '../api'
import type { SnakeSnapshot } from '../types'

defineProps<{
  snapshot: SnakeSnapshot
  running: boolean
}>()

const emit = defineEmits<{
  error: [string]
}>()

async function start() {
  try {
    await api.watchStart()
  } catch (err) {
    emit('error', err instanceof Error ? err.message : String(err))
  }
}

async function stop() {
  try {
    await api.watchStop()
  } catch (err) {
    emit('error', err instanceof Error ? err.message : String(err))
  }
}
</script>

<template>
  <section class="panel">
    <SnakeBoard :snapshot="snapshot" />
    <aside>
      <p class="hint">
        用已训练的 DQN 贪心选择（ε = 0），约 8 帧/秒回放。观看棋盘与训练同为 8×8。先训练再看。
      </p>
      <dl>
        <div>
          <dt>分数</dt>
          <dd>{{ snapshot.score }}</dd>
        </div>
        <div>
          <dt>长度</dt>
          <dd>{{ snapshot.snake.length }}</dd>
        </div>
      </dl>
      <div class="row">
        <button class="primary" type="button" :disabled="running" @click="start">开始观看</button>
        <button type="button" :disabled="!running" @click="stop">停止</button>
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
  min-width: 220px;
}

.hint {
  color: #8b97a8;
  font-size: 13px;
  line-height: 1.55;
}

dl {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

dt {
  color: #8b97a8;
  font-size: 12px;
}

dd {
  margin: 4px 0 0;
  font-size: 28px;
  font-variant-numeric: tabular-nums;
}

.row {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}
</style>
