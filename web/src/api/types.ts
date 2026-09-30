export type InstanceStatus =
  | 'queued'
  | 'starting'
  | 'ready'
  | 'stopping'
  | 'resetting'
  | 'switching'
  | 'closing'
  | 'deleting'
  | 'closed'
  | 'error'
export type Operation = 'stop' | 'reset' | 'switch' | 'close' | 'restart' | 'delete' | 'archive'
export interface Scene {
  id: string
  name: string
  group: string
  description: string
}
export interface Verification {
  outcome: string
  predicates: { name: string; value: boolean | null; evidence: string }[]
  scene_context: string
}
export interface Message {
  id: string
  role: 'user' | 'assistant' | 'tool' | 'system'
  text: string
  tool?: string
  arguments?: string
  status?: string
  duration?: string
  verification?: Verification
}
export interface Plan {
  mission: string
  main_line: {
    id: number
    subgoal: string
    done_criterion: string
    status: string
    retries: number
  }[]
  pointer: number
  branch_stack?: { id: string; subgoal: string; done_criterion: string }[]
  error?: string
}
export interface SnapshotFile<T> {
  data: T
  age_s: number | null
  error: string | null
}
export interface RobotState {
  type?: string
  eef_pose?: {
    position: { x: number; y: number; z: number }
    orientation_euler: { roll: number; pitch: number; yaw: number }
  }
  gripper_qpos?: number[]
  success?: boolean
  done?: boolean
  control?: { controller: string; frequency_hz: number }
}
export interface Instance {
  id: string
  title: string
  sceneId: string
  initialSceneId: string
  sessionId: string
  openedAt: number
  createdAt: number
  status: InstanceStatus
  runStatus: 'idle' | 'running' | 'stopping' | 'completed' | 'cancelled' | 'failed' | 'timed_out'
  archived: boolean
  model: string
  messages: Message[]
  workspace: string
  operation?: string | null
  error?: string | null
  controller: { pid: number; created: number } | null
  snapshot: {
    plan: Plan
    robot: SnapshotFile<{ robots?: Record<string, RobotState> }>
    actions: SnapshotFile<{
      actions?: { id: string; action_type: string; status: string; result?: string }[]
    }>
    observation: SnapshotFile<{
      revision?: number
      reference_view?: string
      views?: { name: string; image_path: string; width: number; height: number }[]
    }> & { updatedAt: number | null; summary: { status: string } }
  }
  verification: Verification | null
  result: { run_status: string; finish_reason: string; error?: { message: string } | null } | null
  usage: Record<string, number>
  durationMs: number
}
export interface Workspace {
  instances: Instance[]
  selectedId: string | null
  collapsed: boolean
  ratio: number
  notice: string
  connected: boolean
}
export interface Settings {
  models: string[]
  defaultModel: string
  driver: string
  dataDirectory: string
  python: string
}
export interface SetupField {
  name: 'model' | 'api_base' | 'api_key'
  title: string
  description: string
  default: string
  required: boolean
  password: boolean
}
export interface SetupProvider {
  name: string
  label: string
  category: string
  isOauth: boolean
  fields: SetupField[]
}
export interface SetupStatus {
  required: boolean
  provider: string
  providers: SetupProvider[]
}
export interface Health {
  name: string
  status: string
  url?: string
  error?: string
  detail?: string
}
export const statusLabels: Record<InstanceStatus, string> = {
  queued: 'Waiting for a slot',
  starting: 'Starting',
  ready: 'Scene ready',
  stopping: 'Stopping',
  resetting: 'Resetting',
  switching: 'Switching',
  closing: 'Closing',
  deleting: 'Deleting',
  closed: 'Closed',
  error: 'Error',
}
