import { useEffect, useRef, useState } from 'react'
import { Camera, ChevronDown, Expand, LoaderCircle, Power, Square } from 'lucide-react'
import type { Instance } from '../../api/types'
import { statusLabels } from '../../api/types'

export default function SceneViewport({
  instance,
  camera,
  onCamera,
  onRestart,
  onStop,
  onError,
}: {
  instance: Instance
  camera?: string
  onCamera: (camera: string) => void
  onRestart: () => void
  onStop: () => void
  onError: (text: string) => void
}) {
  const viewport = useRef<HTMLElement>(null)
  const [fullscreen, setFullscreen] = useState(false)
  const [failedImage, setFailedImage] = useState<string | null>(null)
  const [visible, setVisible] = useState(() => !document.hidden)
  const [frame, setFrame] = useState<{ stream: string; url: string } | null>(null)
  const connection = useRef<WebSocket | null>(null)
  useEffect(() => {
    const changed = () => setFullscreen(document.fullscreenElement === viewport.current)
    const visibilityChanged = () => setVisible(!document.hidden)
    document.addEventListener('fullscreenchange', changed)
    document.addEventListener('visibilitychange', visibilityChanged)
    return () => {
      document.removeEventListener('fullscreenchange', changed)
      document.removeEventListener('visibilitychange', visibilityChanged)
    }
  }, [])
  const observation = instance.snapshot.observation
  const views = observation.data.views || []
  const selected = views.some((view) => view.name === camera)
    ? camera
    : observation.data.reference_view
  const image = selected
    ? `/api/instances/${instance.id}/image?camera=${encodeURIComponent(selected)}&v=${observation.updatedAt}`
    : ''
  const busy = !!instance.operation || !['ready', 'closed', 'error'].includes(instance.status)
  const stream =
    selected && visible && instance.status === 'ready' && !busy
      ? `/api/instances/${instance.id}/image-stream?camera=${encodeURIComponent(selected)}&session=${encodeURIComponent(instance.sessionId)}`
      : ''
  const source = stream && frame?.stream === stream ? frame.url : image
  useEffect(() => {
    setFrame(null)
    if (!stream) return
    let active = true
    let timer: number | undefined
    let currentUrl: string | null = null
    const connect = () => {
      const url = new URL(stream, window.location.href)
      url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
      const socket = new WebSocket(url)
      connection.current = socket
      socket.onopen = () => socket.send('next')
      socket.onmessage = (event) => {
        if (!active) return
        if (typeof event.data === 'string') {
          socket.send('next')
          return
        }
        const next = URL.createObjectURL(event.data)
        setFrame({ stream, url: next })
        if (currentUrl) URL.revokeObjectURL(currentUrl)
        currentUrl = next
      }
      socket.onerror = () => socket.close()
      socket.onclose = () => {
        if (active) timer = window.setTimeout(connect, 1000)
      }
    }
    connect()
    return () => {
      active = false
      window.clearTimeout(timer)
      connection.current?.close()
      connection.current = null
      if (currentUrl) URL.revokeObjectURL(currentUrl)
    }
  }, [stream])
  const robot = Object.values(instance.snapshot.robot.data.robots || {})[0]
  const currentStep = instance.snapshot.plan.main_line.find(
    (step) => step.id === instance.snapshot.plan.pointer,
  )

  return (
    <section ref={viewport} className="scene-viewport real-scene" aria-label="Scene view">
      <div className="viewport-toolbar">
        <label className="camera-select">
          <Camera size={15} />
          <select
            aria-label="Camera view"
            value={selected || ''}
            disabled={!views.length}
            onChange={(event) => onCamera(event.target.value)}
          >
            {!views.length && <option value="">No observations</option>}
            {views.map((view) => (
              <option key={view.name} value={view.name}>
                {view.name}
              </option>
            ))}
          </select>
          <ChevronDown size={13} />
        </label>
        <div className="viewport-tools">
          <button
            className="icon-button"
            aria-label="Save screenshot"
            disabled={!image}
            onClick={async () => {
              try {
                const response = await fetch(image)
                if (!response.ok) throw new Error('Camera image is unavailable')
                const url = URL.createObjectURL(await response.blob())
                const link = document.createElement('a')
                link.href = url
                link.download = `${instance.id}-${selected}.png`
                link.click()
                window.setTimeout(() => URL.revokeObjectURL(url), 1000)
              } catch (error) {
                onError(String(error))
              }
            }}
          >
            <Camera size={16} />
          </button>
          <button
            className="icon-button"
            aria-label="Fullscreen"
            onClick={async () => {
              try {
                if (document.fullscreenElement) await document.exitFullscreen()
                else await viewport.current!.requestFullscreen()
              } catch (error) {
                onError(String(error))
              }
            }}
          >
            <Expand size={16} />
          </button>
        </div>
      </div>
      <div className="scene-image">
        {image && (
          <img
            className="camera-image"
            src={source}
            alt={`${selected} observation for ${instance.title}`}
            onError={() => {
              setFailedImage(source)
              if (source === frame?.url) connection.current?.close()
            }}
            onLoad={() => {
              setFailedImage(null)
              if (source === frame?.url && connection.current?.readyState === WebSocket.OPEN)
                connection.current.send('next')
            }}
          />
        )}
        {(!image || failedImage === source || busy || instance.status !== 'ready') && (
          <div
            className={`scene-overlay ${instance.status === 'closed' && image ? 'historical-overlay' : ''}`}
          >
            <div className="scene-overlay-content">
              {busy && <LoaderCircle className="spin" size={26} />}
              <h2>
                {busy || instance.status !== 'ready'
                  ? statusLabels[instance.status]
                  : 'Waiting for observation'}
              </h2>
              <p>
                {instance.error ||
                  observation.error ||
                  (failedImage === source
                    ? 'Camera image is not available.'
                    : instance.status === 'closed'
                      ? 'Scene closed. This is its last saved observation.'
                      : 'Waiting for this workspace to publish camera images.')}
              </p>
              {!busy && ['closed', 'error'].includes(instance.status) && (
                <button className="button" onClick={onRestart}>
                  <Power size={15} />
                  Restart scene
                </button>
              )}
            </div>
          </div>
        )}
      </div>
      <footer className="viewport-footer">
        <span>
          <i className={`status-dot ${instance.status === 'ready' ? '' : 'off'}`} />
          {instance.status === 'closed'
            ? 'Last saved frame'
            : observation.summary.status !== 'ready'
              ? observation.summary.status
              : (observation.age_s ?? 0) > 3
                ? 'No new frames'
                : 'Observation updated'}
          <span className="footer-divider" />
          {observation.updatedAt
            ? new Date(observation.updatedAt / 1e6).toLocaleTimeString('en-GB')
            : 'No frame'}
        </span>
        <span>
          {robot?.type || 'Robot unknown'}
          <span className="footer-divider" />#{observation.data.revision ?? '—'}
        </span>
      </footer>
      {fullscreen && (
        <div className="fullscreen-controls">
          <span>{currentStep?.subgoal || instance.runStatus}</span>
          {instance.runStatus === 'running' && (
            <button className="button" onClick={onStop}>
              <Square size={14} />
              Stop execution
            </button>
          )}
          {instance.runStatus === 'stopping' && <span>Stopping…</span>}
        </div>
      )}
    </section>
  )
}
