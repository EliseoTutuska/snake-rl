<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import SnakeBoard from './SnakeBoard.vue'
import * as api from '../api'
import type { SnakeSnapshot } from '../types'

const props = defineProps<{
  snapshot: SnakeSnapshot
  active: boolean
}>()

const emit = defineEmits<{
  'update:snapshot': [SnakeSnapshot]
  error: [string]
}>()

const wantedDir = ref(1)
const playing = ref(false)
let timer: number | null = null

async function reset() {
  try {
    const state = await api.playReset()
    emit('update:snapshot', state.snapshot)
    wantedDir.value = state.snapshot.dir
    startLoop()
  } catch (err) {
    emit('error', err instanceof Error ? err.message : String(err))
  }
}

function startLoop() {
  stopLoop()
  playing.value = true
  timer = window.setInterval(tick, 120)
}

function stopLoop() {
  playing.value = false
  if (timer !== null) {
    window.clearInterval(timer)
    timer = null
  }
}

async function tick() {
  if (props.snapshot.done) {
    stopLoop()
    return
  }
  try {
    const state = await api.playStep({ dir: wantedDir.value })
    emit('update:snapshot', state.snapshot)
    if (state.done) stopLoop()
  } catch (err) {
    stopLoop()
    emit('error', err instanceof Error ? err.message : String(err))
  }
}

function onKey(event: KeyboardEvent) {
  if (!props.active) return
  const map: Record<string, number> = {
    ArrowUp: 0,
    KeyW: 0,
    ArrowRight: 1,
    KeyD: 1,
    ArrowDown: 2,
    KeyS: 2,
    ArrowLeft: 3,
    KeyA: 3,
  }
  const dir = map[event.code]
  if (dir === undefined) return
  event.preventDefault()
  wantedDir.value = dir
}

watch(
  () => props.active,
  (active) => {
    if (active && !props.snapshot.done) startLoop()
    else stopLoop()
  },
)

onMounted(async () => {
  window.addEventListener('keydown', onKey)
  try {
    const state = await api.playState()
    emit('update:snapshot', state.snapshot)
    wantedDir.value = state.snapshot.dir
    if (props.active) startLoop()
  } catch (err) {
    emit('error', err instanceof Error ? err.message : String(err))
  }
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKey)
  stopLoop()
})
</script>

<template>
  <section class="panel">
    <SnakeBoard :snapshot="snapshot" />
    <aside>
      <dl>
        <div>
          <dt>分数</dt>
          <dd>{{ snapshot.score }}</dd>
        </div>
        <div>
          <dt>长度</dt>
          <dd>{{ snapshot.snake.length }}</dd>
        </div>
        <div>
          <dt>步数</dt>
          <dd>{{ snapshot.steps }}</dd>
        </div>
      </dl>
      <p class="hint">方向键或 WASD 转向。页面只负责按键，每一步由 Python 环境结算。</p>
      <div class="row">
        <button class="primary" type="button" @click="reset">重新开始</button>
        <button v-if="playing" type="button" @click="stopLoop">暂停</button>
        <button v-else type="button" :disabled="snapshot.done" @click="startLoop">继续</button>
      </div>
      <p v-if="snapshot.done" class="dead">撞了（{{ snapshot.lastReason ?? '结束' }}）。点重新开始。</p>
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
  font-size: 28px;
  font-variant-numeric: tabular-nums;
}

.hint,
.dead {
  color: #8b97a8;
  font-size: 13px;
  line-height: 1.55;
}

.dead {
  color: #fb7185;
}

.row {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}
</style>
