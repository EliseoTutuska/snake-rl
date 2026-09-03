<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ scores: number[] }>()

const points = computed(() => {
  const scores = props.scores
  if (scores.length === 0) return ''
  const max = Math.max(1, ...scores)
  const n = scores.length
  return scores
    .map((s, i) => {
      const x = n === 1 ? 0 : (i / (n - 1)) * 200
      const y = 58 - (s / max) * 52
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
})
</script>

<template>
  <svg class="chart" viewBox="0 0 200 64" preserveAspectRatio="none" aria-hidden="true">
    <polyline v-if="points" :points="points" />
  </svg>
</template>

<style scoped>
.chart {
  width: 100%;
  height: 72px;
  background: #141820;
  border: 1px solid #2a3344;
  border-radius: 10px;
}

polyline {
  fill: none;
  stroke: #2dd4bf;
  stroke-width: 2;
  stroke-linejoin: round;
  stroke-linecap: round;
}
</style>
