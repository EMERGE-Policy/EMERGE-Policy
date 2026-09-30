import { useEffect, useRef, useState } from 'react'
import {
  ArrowUp,
  ArrowUpRight,
  Box,
  ChevronDown,
  CircleHelp,
  FileText,
  LoaderCircle,
  Monitor,
  Plus,
  Power,
  RotateCcw,
  Settings2,
  Square,
  X,
} from 'lucide-react'
import type { Health, Instance, Operation, Scene, Settings, SetupStatus } from './api/types'
import { statusLabels } from './api/types'
import { api } from './api/client'
import { displayTitle } from './presentation'
import { useWorkspace } from './state/workspace'
import InstanceSidebar, { type InstanceCommand } from './components/instances/InstanceSidebar'
import ScenePicker from './components/scene/ScenePicker'
import SceneViewport from './components/scene/SceneViewport'
import Conversation from './components/chat/Conversation'
import Dialog from './components/ui/Dialog'
import SetupWizard from './components/setup/SetupWizard'

type Modal =
  | { type: 'picker'; id?: string }
  | { type: 'confirm'; id: string; operation: Operation }
  | { type: 'rename'; id: string }
  | { type: 'settings' }
  | { type: 'files'; id: string }
  | null

export default function App() {
  const [setup, setSetup] = useState<SetupStatus | null>(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let active = true
    api<SetupStatus>('/setup')
      .then((status) => {
        if (active) setSetup(status)
      })
      .catch((error) => {
        if (active) setError(String(error))
      })
    return () => {
      active = false
    }
  }, [attempt])

  if (!setup) {
    return (
      <main className="setup-loading" aria-live="polite">
        {error ? (
          <>
            <p role="alert" className="error-text">
              {error}
            </p>
            <button
              className="button secondary"
              onClick={() => {
                setError('')
                setAttempt((value) => value + 1)
              }}
            >
              Retry
            </button>
          </>
        ) : (
          <>
            <LoaderCircle className="spin" size={22} />
            <p>Checking workspace configuration…</p>
          </>
        )}
      </main>
    )
  }
  return setup.required ? <SetupWizard status={setup} onComplete={setSetup} /> : <WorkspaceApp />
}

function WorkspaceApp() {
  const { state, preferences, setPreferences, error, setError } = useWorkspace()
  const [modal, setModal] = useState<Modal>(null)
  const [rename, setRename] = useState('')
  const [scenes, setScenes] = useState<Scene[]>([])
  const [settings, setSettings] = useState<Settings | null>(null)
  const [health, setHealth] = useState<Health[]>([])
  const [checkingHealth, setCheckingHealth] = useState(false)
  const [pending, setPending] = useState(false)
  const [files, setFiles] = useState<{ path: string; size: number }[]>([])
  const [logs, setLogs] = useState('')
  const panels = useRef<HTMLDivElement>(null)
  const modelInput = useRef<HTMLInputElement>(null)
  const current = state.instances.find((item) => item.id === state.selectedId)
  const scene = scenes.find((item) => item.id === current?.sceneId)
  const running = current?.runStatus === 'running'
  const stopping = current?.runStatus === 'stopping'
  const canSend =
    !!current &&
    state.connected &&
    current.status === 'ready' &&
    !current.operation &&
    !running &&
    !stopping &&
    !pending
  const draft = current ? preferences.drafts[current.id] || '' : ''
  const modalInstance =
    modal && 'id' in modal ? state.instances.find((item) => item.id === modal.id) : undefined
  const activeCount = state.instances.filter(
    (item) => item.controller || item.status === 'starting',
  ).length
  const plan = current?.snapshot.plan
  const robot = current && Object.values(current.snapshot.robot.data.robots || {})[0]
  const focus =
    plan?.branch_stack?.at(-1)?.subgoal ||
    plan?.main_line.find((step) => step.id === plan.pointer)?.subgoal

  useEffect(() => {
    if (!state.connected) return
    Promise.all([api<{ scenes: Scene[] }>('/catalog'), api<Settings>('/settings')])
      .then(([catalog, config]) => {
        setScenes(catalog.scenes)
        setSettings(config)
      })
      .catch((error) => setError(String(error)))
  }, [state.connected, setError])

  async function operate(item: Instance, operation: Operation, sceneId?: string) {
    setPending(true)
    try {
      await api(`/instances/${item.id}/operations`, 'POST', { operation, sceneId })
      setModal(null)
    } catch (error) {
      setError(String(error))
    } finally {
      setPending(false)
    }
  }

  function command(item: Instance, action: InstanceCommand) {
    if (action === 'rename') {
      setRename(item.title)
      setModal({ type: 'rename', id: item.id })
    } else if (action === 'restore')
      api(`/instances/${item.id}`, 'PATCH', { archived: false }).catch((error) =>
        setError(String(error)),
      )
    else setModal({ type: 'confirm', id: item.id, operation: action })
  }

  async function send() {
    if (!current || !canSend || !draft.trim()) return
    setPending(true)
    const key = current.id
    try {
      await api(`/instances/${key}/runs`, 'POST', {
        message: draft,
        model: modelInput.current?.value.trim() || current.model,
      })
      setPreferences((value) => ({
        ...value,
        drafts: { ...value.drafts, [key]: value.drafts[key] === draft ? '' : value.drafts[key] },
      }))
    } catch (error) {
      setError(String(error))
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="app-shell">
      <InstanceSidebar
        state={state}
        onSelect={(id) => setPreferences((value) => ({ ...value, selectedId: id }))}
        onNew={() => setModal({ type: 'picker' })}
        onCollapse={() => setPreferences((value) => ({ ...value, collapsed: !value.collapsed }))}
        onSettings={() => setModal({ type: 'settings' })}
        onCommand={command}
      />
      <main className="workspace">
        <header className="workspace-header">
          <div className="workspace-title">
            <h1 title={current?.title}>
              {current ? displayTitle(current.title) : 'EMERGE Policy'}
            </h1>
            <div className="header-meta">
              {current && (
                <span className="instance-status">
                  <i
                    className={`status-dot ${current.status === 'closed' ? 'off' : current.status === 'error' ? 'failed' : ''}`}
                  />
                  {statusLabels[current.status]}
                </span>
              )}
              <span>{current ? 'Robot workspace' : 'Your robot workspace'}</span>
            </div>
          </div>
          <div className="header-actions">
            <span className="active-count">{activeCount} / 10 instances</span>
            {current && (
              <>
                <button
                  className="icon-button"
                  aria-label="Logs and artifacts"
                  onClick={async () => {
                    setModal({ type: 'files', id: current.id })
                    setFiles([])
                    setLogs('Loading…')
                    try {
                      const [listing, log] = await Promise.all([
                        api<{ path: string; size: number }[]>(`/instances/${current.id}/files`),
                        api<{ text: string }>(`/instances/${current.id}/logs`),
                      ])
                      setFiles(listing)
                      setLogs(log.text)
                    } catch (error) {
                      setError(String(error))
                      setLogs('Unable to load logs.')
                    }
                  }}
                >
                  <FileText size={17} />
                </button>
                <details className="menu scene-menu">
                  <summary className="button secondary">
                    <Settings2 size={15} />
                    <span>Scene controls</span>
                    <ChevronDown size={13} />
                  </summary>
                  <div
                    className="menu-popover"
                    onClick={(event) =>
                      event.currentTarget.closest('details')?.removeAttribute('open')
                    }
                  >
                    <button
                      disabled={!current.controller || !!current.operation || !state.connected}
                      onClick={() =>
                        setModal({ type: 'confirm', id: current.id, operation: 'reset' })
                      }
                    >
                      <RotateCcw size={15} />
                      Reset scene
                    </button>
                    <button
                      disabled={!current.controller || !!current.operation || !state.connected}
                      onClick={() => setModal({ type: 'picker', id: current.id })}
                    >
                      <Box size={15} />
                      Switch scene
                    </button>
                    <hr />
                    <button
                      disabled={!!current.operation || !state.connected}
                      onClick={() =>
                        setModal({
                          type: 'confirm',
                          id: current.id,
                          operation:
                            current.status === 'closed' || !current.controller
                              ? 'restart'
                              : 'close',
                        })
                      }
                    >
                      <Power size={15} />
                      {current.status === 'closed' || !current.controller
                        ? 'Restart scene'
                        : 'Close scene'}
                    </button>
                  </div>
                </details>
              </>
            )}
          </div>
        </header>
        {!state.connected && (
          <div className="notice" role="status">
            <LoaderCircle className="spin" size={15} />
            Connecting to the EMERGE backend. Tasks continue on the server.
          </div>
        )}
        {(error || state.notice || current?.error) && (
          <div className="notice" role="status">
            <CircleHelp size={16} />
            <span>{error || current?.error || state.notice}</span>
            <button
              className="icon-button"
              aria-label="Dismiss notice"
              onClick={() => setError('')}
            >
              <X size={15} />
            </button>
          </div>
        )}
        {current ? (
          <>
            <div
              className="workspace-panels"
              ref={panels}
              style={{ '--scene-ratio': `${state.ratio}%` } as React.CSSProperties}
            >
              <div className="scene-panel">
                <div className="scene-heading">
                  <span>
                    <Box size={14} />
                    Scene observation
                  </span>
                  <span className="scene-source" title={current.sceneId}>
                    {scene?.group || settings?.driver}
                  </span>
                </div>
                <SceneViewport
                  key={current.id}
                  instance={current}
                  camera={preferences.cameras[current.id]}
                  onCamera={(camera) =>
                    setPreferences((value) => ({
                      ...value,
                      cameras: { ...value.cameras, [current.id]: camera },
                    }))
                  }
                  onRestart={() =>
                    setModal({ type: 'confirm', id: current.id, operation: 'restart' })
                  }
                  onStop={() => operate(current, 'stop')}
                  onError={setError}
                />
                <div className="robot-strip">
                  <span>
                    <small>Environment success</small>
                    <strong>
                      {robot?.success === undefined ? 'Unknown' : String(robot.success)}
                    </strong>
                  </span>
                  <span>
                    <small>Gripper position</small>
                    <strong>
                      {robot?.gripper_qpos?.map((value) => value.toFixed(3)).join(', ') ||
                        'Unknown'}
                    </strong>
                  </span>
                  <span title={JSON.stringify(robot?.eef_pose)}>
                    <small>End-effector position</small>
                    <strong>
                      {robot?.eef_pose
                        ? Object.values(robot.eef_pose.position)
                            .map((value) => value.toFixed(3))
                            .join(', ') + ' m'
                        : 'Unknown'}
                    </strong>
                  </span>
                </div>
              </div>
              <div
                className="panel-divider"
                role="separator"
                aria-label="Resize scene and conversation"
                aria-orientation="vertical"
                aria-valuemin={45}
                aria-valuemax={80}
                aria-valuenow={Math.round(state.ratio)}
                tabIndex={0}
                onPointerDown={(event) => {
                  event.currentTarget.setPointerCapture(event.pointerId)
                  event.currentTarget.dataset.dragging = 'true'
                }}
                onPointerMove={(event) => {
                  if (event.currentTarget.dataset.dragging !== 'true') return
                  const rect = panels.current!.getBoundingClientRect()
                  const ratio = Math.min(
                    80,
                    Math.max(45, ((event.clientX - rect.left) / rect.width) * 100),
                  )
                  setPreferences((value) => ({ ...value, ratio }))
                }}
                onPointerUp={(event) => event.currentTarget.releasePointerCapture(event.pointerId)}
                onLostPointerCapture={(event) => {
                  delete event.currentTarget.dataset.dragging
                }}
                onKeyDown={(event) => {
                  if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
                    event.preventDefault()
                    setPreferences((value) => ({
                      ...value,
                      ratio:
                        event.key === 'Home'
                          ? 45
                          : event.key === 'End'
                            ? 80
                            : Math.min(
                                80,
                                Math.max(45, value.ratio + (event.key === 'ArrowLeft' ? -2 : 2)),
                              ),
                    }))
                  }
                }}
              >
                <span />
              </div>
              <Conversation key={current.sessionId} instance={current} />
            </div>
            <section className="composer-area" aria-label="Instruction composer">
              <div className="task-strip">
                <span>
                  {running || stopping ? (
                    <LoaderCircle size={14} className="spin" />
                  ) : (
                    <span className="tiny-orbit" />
                  )}
                  {stopping
                    ? 'Stopping · Awaiting robot acknowledgement'
                    : running
                      ? focus || 'Agent running'
                      : current.runStatus === 'idle'
                        ? 'Ready for an instruction'
                        : `Execution ${current.runStatus}`}
                </span>
                {(!!plan?.main_line.length || current.verification) && (
                  <span className="task-summary">
                    {!!plan?.main_line.length && (
                      <span>
                        {plan.main_line.filter((step) => step.status === 'done').length} /{' '}
                        {plan.main_line.length} steps
                      </span>
                    )}
                    <span>Verification: {current.verification?.outcome || 'not verified'}</span>
                  </span>
                )}
              </div>
              <form
                className="composer"
                onSubmit={(event) => {
                  event.preventDefault()
                  void send()
                }}
              >
                <textarea
                  rows={2}
                  aria-label="Instruction"
                  placeholder={
                    running
                      ? 'Draft your next instruction…'
                      : 'Describe what you want the robot to do…'
                  }
                  value={draft}
                  onChange={(event) => {
                    const text = event.target.value
                    setPreferences((value) => ({
                      ...value,
                      drafts: { ...value.drafts, [current.id]: text },
                    }))
                  }}
                  onKeyDown={(event) => {
                    if (
                      event.key === 'Enter' &&
                      !event.shiftKey &&
                      !event.nativeEvent.isComposing
                    ) {
                      event.preventDefault()
                      void send()
                    }
                  }}
                />
                <div className="composer-toolbar">
                  <label className="model-select">
                    <span className="model-orbit" />
                    <input
                      key={current.id + current.model}
                      ref={modelInput}
                      aria-label="Main agent model"
                      list="model-options"
                      defaultValue={current.model}
                      disabled={!!current.operation}
                      onBlur={(event) => {
                        const model = event.target.value.trim()
                        if (model && model !== current.model)
                          api(`/instances/${current.id}`, 'PATCH', { model }).catch((error) =>
                            setError(String(error)),
                          )
                      }}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          event.preventDefault()
                          event.currentTarget.blur()
                        }
                      }}
                    />
                    <datalist id="model-options">
                      {settings?.models.map((model) => (
                        <option key={model} value={model} />
                      ))}
                    </datalist>
                  </label>
                  <div className="composer-right">
                    <span>{running ? 'Draft kept for next turn' : 'Enter to send'}</span>
                    {running || stopping ? (
                      <button
                        className="send-button"
                        type="button"
                        disabled={pending || !!current.operation || !state.connected}
                        aria-label={stopping ? 'Retry stop' : 'Stop execution'}
                        onClick={() => operate(current, 'stop')}
                      >
                        {stopping ? (
                          <LoaderCircle className="spin" size={18} />
                        ) : (
                          <Square size={16} fill="currentColor" />
                        )}
                      </button>
                    ) : (
                      <button
                        className="send-button"
                        type="submit"
                        aria-label="Send instruction"
                        disabled={!canSend || !draft.trim()}
                      >
                        <ArrowUp size={20} />
                      </button>
                    )}
                  </div>
                </div>
              </form>
              <div className="composer-caption">
                <span>
                  {current.usage.total_tokens || 0} main-agent tokens ·{' '}
                  {(current.durationMs / 1000).toFixed(1)}s
                </span>
                <span>Shift + Enter for a new line</span>
              </div>
            </section>
          </>
        ) : (
          <div className="workspace-empty">
            <img src="/emerge-logo.png" alt="" />
            <span className="eyebrow">A ROBOT MIND EMERGES</span>
            <h2>Start with a scene.</h2>
            <p>Create a conversation to launch its own robot environment and workspace.</p>
            <button
              className="button primary"
              disabled={!state.connected || !scenes.length}
              onClick={() => setModal({ type: 'picker' })}
            >
              <Plus size={17} />
              New conversation
              <ArrowUpRight size={16} />
            </button>
          </div>
        )}
      </main>

      {modal?.type === 'picker' && (
        <ScenePicker
          scenes={scenes}
          pending={pending}
          switching={!!modal.id}
          currentScene={modalInstance?.sceneId}
          atCapacity={activeCount >= 10}
          onClose={() => setModal(null)}
          onSelect={async (sceneId) => {
            if (modalInstance) {
              await operate(modalInstance, 'switch', sceneId)
              return
            }
            setPending(true)
            try {
              const item = await api<Instance>('/instances', 'POST', { sceneId })
              setPreferences((value) => ({ ...value, selectedId: item.id }))
              setModal(null)
            } catch (error) {
              setError(String(error))
            } finally {
              setPending(false)
            }
          }}
        />
      )}
      {modal?.type === 'rename' && modalInstance && (
        <Dialog title="Rename instance" onClose={() => setModal(null)}>
          <form
            onSubmit={async (event) => {
              event.preventDefault()
              try {
                await api(`/instances/${modalInstance.id}`, 'PATCH', { title: rename.trim() })
                setModal(null)
              } catch (error) {
                setError(String(error))
              }
            }}
          >
            <label className="form-label">
              Instance name
              <input
                autoFocus
                maxLength={200}
                value={rename}
                onChange={(event) => setRename(event.target.value)}
              />
            </label>
            <footer className="dialog-footer">
              <button type="button" className="button secondary" onClick={() => setModal(null)}>
                Cancel
              </button>
              <button className="button primary" disabled={!rename.trim()}>
                Save
              </button>
            </footer>
          </form>
        </Dialog>
      )}
      {modal?.type === 'confirm' && modalInstance && (
        <Dialog
          title={`${modal.operation[0].toUpperCase() + modal.operation.slice(1)} this scene?`}
          onClose={() => setModal(null)}
        >
          <p className="confirm-title">{modalInstance.title}</p>
          <p className="dialog-description">
            {modal.operation === 'delete'
              ? 'Stop the task, close its Controller, and permanently delete this instance, conversation, logs, and artifacts.'
              : modal.operation === 'reset'
                ? `Stop execution, clear the current conversation and task workspace, and reload the startup scene: ${modalInstance.initialSceneId}.`
                : modal.operation === 'restart'
                  ? 'Load a fresh environment in this instance. Previous physical state will not be restored; a new conversation will begin.'
                  : 'Stop execution and release this scene. Its current conversation and retained artifacts remain available.'}
          </p>
          <footer className="dialog-footer">
            <button className="button secondary" onClick={() => setModal(null)}>
              Cancel
            </button>
            <button
              className={`button ${modal.operation === 'delete' ? 'destructive' : 'primary'}`}
              disabled={pending}
              onClick={() => operate(modalInstance, modal.operation)}
            >
              Confirm {modal.operation === 'delete' ? 'deletion' : modal.operation}
            </button>
          </footer>
        </Dialog>
      )}
      {modal?.type === 'settings' && (
        <Dialog title="Settings" onClose={() => setModal(null)}>
          <section className="settings-section">
            <h3>
              <Monitor size={17} />
              Appearance
            </h3>
            <div className="settings-row">
              <span>Theme</span>
              <span>System</span>
            </div>
          </section>
          <section className="settings-section">
            <h3>Runtime</h3>
            <p className="muted">
              Driver: {settings?.driver}
              <br />
              Python: {settings?.python}
              <br />
              Data: {settings?.dataDirectory}
            </p>
            {current && (
              <p className="muted">
                Workspace: {current.workspace}
                <br />
                Session: {current.sessionId}
                <br />
                Controller PID: {current.controller?.pid || 'not running'}
              </p>
            )}
          </section>
          <section className="settings-section">
            <h3>Model services</h3>
            <button
              className="button secondary"
              disabled={checkingHealth}
              onClick={async () => {
                setCheckingHealth(true)
                try {
                  setHealth(await api<Health[]>('/health'))
                } catch (error) {
                  setError(String(error))
                } finally {
                  setCheckingHealth(false)
                }
              }}
            >
              {checkingHealth ? 'Checking…' : 'Check services'}
            </button>
            {!health.length && (
              <p className="muted">
                Not probed. Uses the existing Emerge service discovery configuration.
              </p>
            )}
            {health.map((service, index) => (
              <div className="service-health" key={index}>
                <strong>
                  {service.name} · {service.status}
                </strong>
                <p>{service.url}</p>
                <p>{service.error || service.detail}</p>
              </div>
            ))}
          </section>
        </Dialog>
      )}
      {modal?.type === 'files' && modalInstance && (
        <Dialog title="Logs and artifacts" onClose={() => setModal(null)} wide>
          <div className="log-controls">
            {['agent', 'controller'].map((kind) => (
              <button
                className="button secondary"
                key={kind}
                onClick={async () => {
                  try {
                    setLogs(
                      (
                        await api<{ text: string }>(
                          `/instances/${modalInstance.id}/logs?kind=${kind}`,
                        )
                      ).text,
                    )
                  } catch (error) {
                    setError(String(error))
                  }
                }}
              >
                {kind} log
              </button>
            ))}
            <button
              className="button secondary"
              onClick={() => {
                const url = URL.createObjectURL(
                  new Blob(
                    [
                      modalInstance.messages
                        .map(
                          (message) =>
                            `${message.role.toUpperCase()}${message.tool ? ' · ' + message.tool : ''}\n${message.arguments || ''}\n${message.text}`,
                        )
                        .join('\n\n'),
                    ],
                    { type: 'text/markdown' },
                  ),
                )
                const link = document.createElement('a')
                link.href = url
                link.download = 'conversation.md'
                link.click()
                window.setTimeout(() => URL.revokeObjectURL(url), 1000)
              }}
            >
              Export conversation
            </button>
          </div>
          <pre className="log-view">{logs}</pre>
          <div className="artifact-list">
            {files.map((file) => (
              <a
                key={file.path}
                href={`/api/instances/${modalInstance.id}/file?path=${encodeURIComponent(file.path)}`}
                target="_blank"
                rel="noreferrer"
              >
                {file.path}
                <small>{(file.size / 1024).toFixed(1)} KB</small>
              </a>
            ))}
          </div>
        </Dialog>
      )}
    </div>
  )
}
