<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import PlayPanel from './components/PlayPanel.vue'
import TrainPanel from './components/TrainPanel.vue'
import WatchPanel from './components/WatchPanel.vue'
import { connectWs, trainStatus, watchStatus } from './api'
import { emptySnapshot, emptyTrainStatus, type HubMessage, type TrainStatus } from './types'

type Mode = 'play' | 'train' | 'watch'

const mode = ref<Mode>('play')
const playSnap = ref(emptySnapshot(10))
const watchSnap = ref(emptySnapshot(10))
const watchRunning = ref(false)
const train = ref<TrainStatus>(emptyTrainStatus())
const banner = ref('')
const bannerKind = ref<'error' | 'info'>('info')

let stopWs: (() => void) | null = null

function showError(message: string) {
  bannerKind.value = 'error'
  banner.value = message
}

function showInfo(message: string) {
  bannerKind.value = 'info'
  banner.value = message
}

function onHub(msg: HubMessage) {
  if (msg.channel === 'play' && 'snapshot' in msg && msg.snapshot) {
    playSnap.value = msg.snapshot as typeof playSnap.value
  }
  if (msg.channel === 'train') {
    train.value = msg as TrainStatus
  }
  if (msg.channel === 'watch' && 'snapshot' in msg && msg.snapshot) {
    watchSnap.value = msg.snapshot as typeof watchSnap.value
    if ('running' in msg) watchRunning.value = Boolean(msg.running)
  }
}

onMounted(async () => {
  stopWs = connectWs(onHub, showError)
  try {
    train.value = await trainStatus()
    const watch = await watchStatus()
    watchSnap.value = watch.snapshot
    watchRunning.value = watch.running
  } catch (err) {
    showError(err instanceof Error ? err.message : String(err))
  }
})

onUnmounted(() => stopWs?.())
</script>

<template>
  <div class="page">
    <header>
      <div>
        <p class="eyebrow">Python 环境 · Vue 看板</p>
        <h1>贪吃蛇 RL</h1>
      </div>
      <nav>
        <button class="tab" type="button" :class="{ on: mode === 'play' }" @click="mode = 'play'">
          自己玩
        </button>
        <button class="tab" type="button" :class="{ on: mode === 'train' }" @click="mode = 'train'">
          训练
        </button>
        <button class="tab" type="button" :class="{ on: mode === 'watch' }" @click="mode = 'watch'">
          观看
        </button>
      </nav>
    </header>

    <p v-if="banner" class="banner" :class="bannerKind">{{ banner }}</p>

    <PlayPanel
      v-show="mode === 'play'"
      :snapshot="playSnap"
      :active="mode === 'play'"
      @update:snapshot="playSnap = $event"
      @error="showError"
    />
    <TrainPanel
      v-show="mode === 'train'"
      :status="train"
      @update:status="train = $event"
      @error="showError"
      @info="showInfo"
    />
    <WatchPanel
      v-show="mode === 'watch'"
      :snapshot="watchSnap"
      :running="watchRunning"
      @error="showError"
    />
  </div>
</template>

<style scoped>
.page {
  max-width: 980px;
  margin: 0 auto;
  padding: 28px 20px 64px;
}

header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-end;
  margin-bottom: 24px;
}

.eyebrow {
  margin: 0 0 4px;
  color: #2dd4bf;
  letter-spacing: 0.08em;
  font-size: 12px;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  font-size: 32px;
  font-weight: 650;
}

nav {
  display: flex;
  gap: 6px;
  padding: 4px;
  background: #151b24;
  border: 1px solid #2a3344;
  border-radius: 12px;
}

.tab {
  border: none;
  background: transparent;
  padding: 8px 14px;
  border-radius: 8px;
}

.tab.on {
  background: #1f2a3a;
  color: #fbbf24;
}

.banner {
  margin: 0 0 18px;
  padding: 10px 12px;
  border-radius: 10px;
  font-size: 13px;
}

.banner.error {
  background: #3a1d27;
  color: #fecdd3;
}

.banner.info {
  background: #12352f;
  color: #ccfbf1;
}
</style>
