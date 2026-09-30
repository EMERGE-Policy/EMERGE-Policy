import { useEffect, useState } from 'react'
import type { Workspace } from '../api/types'

const preferenceKey = 'emerge.web.preferences.v1'
interface Preferences {
  selectedId: string | null
  collapsed: boolean
  ratio: number
  drafts: Record<string, string>
  cameras: Record<string, string>
}

export function useWorkspace() {
  const [preferences, setPreferences] = useState<Preferences>(() => {
    const defaults = { selectedId: null, collapsed: false, ratio: 70, drafts: {}, cameras: {} }
    try {
      return { ...defaults, ...JSON.parse(localStorage.getItem(preferenceKey) || '{}') }
    } catch {
      return defaults
    }
  })
  const [remote, setRemote] = useState<Pick<Workspace, 'instances' | 'notice'>>({
    instances: [],
    notice: '',
  })
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    const events = new EventSource('/api/events')
    events.onmessage = (event) => {
      setRemote(JSON.parse(event.data))
      setConnected(true)
    }
    events.onerror = () => setConnected(false)
    return () => events.close()
  }, [])
  useEffect(() => {
    try {
      localStorage.setItem(preferenceKey, JSON.stringify(preferences))
    } catch {
      setError('Browser storage is unavailable. Drafts and layout may not survive refresh.')
    }
  }, [preferences])
  return {
    state: { ...remote, ...preferences, connected } as Workspace,
    preferences,
    setPreferences,
    error,
    setError,
  }
}
