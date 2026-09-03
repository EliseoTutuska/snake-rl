<script setup lang="ts">
import { computed } from 'vue'
import type { SnakeSnapshot } from '../types'

const props = defineProps<{
  snapshot: SnakeSnapshot
  compact?: boolean
}>()

const cells = computed(() => {
  const { gridSize, snake, food } = props.snapshot
  const kinds = Array.from({ length: gridSize * gridSize }, () => 'empty')
  snake.forEach((p, i) => {
    const idx = p.y * gridSize + p.x
    if (idx < 0 || idx >= kinds.length) return
    kinds[idx] = i === 0 ? 'head' : i === snake.length - 1 ? 'tail' : 'body'
  })
  const foodIdx = food.y * gridSize + food.x
  if (foodIdx >= 0 && foodIdx < kinds.length && kinds[foodIdx] === 'empty') {
    kinds[foodIdx] = 'food'
  }
  return kinds.map((kind, i) => ({
    kind,
    i,
  }))
})

const dirClass = computed(() => `dir-${props.snapshot.dir}`)
</script>

<template>
  <div
    class="board-wrap"
    :class="{ compact, dead: snapshot.done }"
    :style="{ '--n': snapshot.gridSize }"
  >
    <div class="board">
      <div
        v-for="cell in cells"
        :key="cell.i"
        class="cell"
        :class="[cell.kind, cell.kind === 'head' ? dirClass : '']"
      />
    </div>
  </div>
</template>

<style scoped>
.board-wrap {
  --n: 10;
  width: min(52vh, 460px);
  aspect-ratio: 1;
}

.board-wrap.compact {
  width: min(42vh, 360px);
}

.board {
  display: grid;
  grid-template-columns: repeat(var(--n), 1fr);
  grid-template-rows: repeat(var(--n), 1fr);
  width: 100%;
  height: 100%;
  background: #141820;
  border: 1px solid #2a3344;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.03);
}

.cell {
  position: relative;
}

.cell.empty {
  background: #141820;
  box-shadow: inset 0 0 0 0.5px rgba(255, 255, 255, 0.03);
}

.cell.body,
.cell.tail {
  background: #2dd4bf;
  border-radius: 22%;
  transform: scale(0.86);
}

.cell.tail {
  opacity: 0.55;
}

.cell.head {
  background: #fbbf24;
  border-radius: 28%;
  transform: scale(0.9);
  z-index: 1;
}

.cell.head::after {
  content: '';
  position: absolute;
  width: 18%;
  height: 18%;
  background: #111827;
  border-radius: 50%;
  top: 28%;
}

.cell.head.dir-0::after {
  left: 22%;
  box-shadow: 28px 0 0 #111827;
}

.cell.head.dir-1::after {
  left: 52%;
  top: 22%;
  box-shadow: 0 28px 0 #111827;
}

.cell.head.dir-2::after {
  left: 22%;
  top: 52%;
  box-shadow: 28px 0 0 #111827;
}

.cell.head.dir-3::after {
  left: 22%;
  top: 22%;
  box-shadow: 0 28px 0 #111827;
}

.cell.food {
  background: transparent;
}

.cell.food::after {
  content: '';
  position: absolute;
  inset: 22%;
  background: #fb7185;
  border-radius: 50%;
  box-shadow: 0 0 10px rgba(251, 113, 133, 0.45);
}

.dead .board {
  filter: saturate(0.65);
}
</style>
