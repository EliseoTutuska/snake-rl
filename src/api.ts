import type { HubMessage, PlayState, TrainStatus, WatchStatus } from './types'

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = (await response.json()) as { detail?: unknown }
      detail =
        typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body)
    } catch {
      detail = await response.text()
    }
    throw new Error(detail || `HTTP ${response.status}`)
  }
  return response.json() as Promise<T>
}

export async function playReset() {
  return json<PlayState>(await fetch('/api/play/reset', { method: 'POST' }))
}

export async function playState() {
  return json<PlayState>(await fetch('/api/play/state'))
}

export async function playStep(body: { dir?: number; action?: number }) {
  return json<PlayState>(
    await fetch('/api/play/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  )
}

export async function trainStart() {
  return json<TrainStatus>(await fetch('/api/train/start', { method: 'POST' }))
}

export async function trainStop() {
  return json<TrainStatus>(await fetch('/api/train/stop', { method: 'POST' }))
}

export async function trainReset() {
  return json<TrainStatus>(await fetch('/api/train/reset', { method: 'POST' }))
}

export async function trainStatus() {
  return json<TrainStatus>(await fetch('/api/train/status'))
}

export async function trainSave() {
  return json<{ ok: boolean; path: string; qSize: number }>(
    await fetch('/api/train/save', { method: 'POST' }),
  )
}

export async function trainLoad() {
  return json<TrainStatus>(await fetch('/api/train/load', { method: 'POST' }))
}

export async function watchStart() {
  return json<WatchStatus>(await fetch('/api/watch/start', { method: 'POST' }))
}

export async function watchStop() {
  return json<WatchStatus>(await fetch('/api/watch/stop', { method: 'POST' }))
}

export async function watchStatus() {
  return json<WatchStatus>(await fetch('/api/watch/status'))
}

export function connectWs(
  onMessage: (msg: HubMessage) => void,
  onError?: (err: string) => void,
): () => void {
  let closed = false
  let socket: WebSocket | null = null
  let timer: number | null = null

  const connect = () => {
    if (closed) return
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    socket = new WebSocket(`${proto}://${location.host}/api/ws`)
    socket.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data) as HubMessage)
      } catch {
        onError?.('无法解析服务器消息')
      }
    }
    socket.onerror = () => onError?.('WebSocket 连接失败，请确认 Python API 已启动')
    socket.onclose = () => {
      if (closed) return
      timer = window.setTimeout(connect, 1500)
    }
  }

  connect()
  return () => {
    closed = true
    if (timer !== null) window.clearTimeout(timer)
    socket?.close()
  }
}
