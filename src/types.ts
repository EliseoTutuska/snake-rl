export interface Point {
  x: number
  y: number
}

export interface SnakeSnapshot {
  gridSize: number
  snake: Point[]
  food: Point
  dir: number
  score: number
  done: boolean
  steps: number
  stepsWithoutFood: number
  lastReward: number
  lastReason?: string | null
}

export interface PlayState {
  observation: number[]
  snapshot: SnakeSnapshot
  reward: number
  done: boolean
  info: {
    score?: number
    length?: number
    reason?: string | null
  }
}

export interface TrainStatus {
  channel: 'train'
  running: boolean
  episode: number
  epsilon: number
  meanScore: number
  totalSteps: number
  scores: number[]
  qSize: number
  lastTdError: number
  snapshot: SnakeSnapshot
  device?: string
  deaths?: {
    wall: number
    self: number
    timeout: number
  }
}

export interface WatchStatus {
  channel: 'watch'
  running: boolean
  snapshot: SnakeSnapshot
}

export type HubMessage =
  | ({ channel: 'play' } & PlayState)
  | TrainStatus
  | WatchStatus
  | { channel: string; [key: string]: unknown }

export function emptySnapshot(gridSize = 10): SnakeSnapshot {
  return {
    gridSize,
    snake: [],
    food: { x: 0, y: 0 },
    dir: 1,
    score: 0,
    done: false,
    steps: 0,
    stepsWithoutFood: 0,
    lastReward: 0,
    lastReason: null,
  }
}

export function emptyTrainStatus(): TrainStatus {
  return {
    channel: 'train',
    running: false,
    episode: 0,
    epsilon: 1,
    meanScore: 0,
    totalSteps: 0,
    scores: [],
    qSize: 0,
    lastTdError: 0,
    device: 'cpu',
    deaths: { wall: 0, self: 0, timeout: 0 },
    snapshot: emptySnapshot(8),
  }
}
