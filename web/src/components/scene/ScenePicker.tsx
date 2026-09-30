import { useState } from 'react'
import { ArrowUpRight, Box, Check, Search } from 'lucide-react'
import type { Scene } from '../../api/types'
import { displayTitle } from '../../presentation'
import Dialog from '../ui/Dialog'

export default function ScenePicker({
  switching,
  currentScene,
  atCapacity,
  onSelect,
  onClose,
  scenes,
  pending = false,
}: {
  switching: boolean
  currentScene?: string
  atCapacity: boolean
  onSelect: (id: string) => void
  onClose: () => void
  scenes: Scene[]
  pending?: boolean
}) {
  const [group, setGroup] = useState('All scenes')
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<string>('')
  const chosen = scenes.find((scene) => scene.id === selected)
  const filtered = scenes.filter(
    (scene) =>
      (group === 'All scenes' || scene.group === group) &&
      `${scene.name} ${scene.id}`.toLowerCase().includes(query.toLowerCase()),
  )
  return (
    <Dialog title={switching ? 'Switch scene' : 'Start a new scene'} onClose={onClose} wide>
      <p className="muted dialog-description">
        {switching
          ? 'Switching stops the current task, clears its context, and starts a new conversation.'
          : 'Choose an environment with its own conversation and workspace.'}
      </p>
      <label className="search-field scene-search">
        <Search size={17} />
        <input
          autoFocus
          placeholder="Search by scene name or filename"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>
      <div className="scene-picker-body">
        <nav className="scene-groups" aria-label="Scene groups">
          {['All scenes', ...new Set(scenes.map((scene) => scene.group))].map((name) => (
            <button
              key={name}
              className={group === name ? 'selected' : ''}
              onClick={() => setGroup(name)}
            >
              {name}
              <span>
                {name === 'All scenes'
                  ? scenes.length
                  : scenes.filter((scene) => scene.group === name).length}
              </span>
            </button>
          ))}
        </nav>
        <div className="scene-options">
          {filtered.map((scene) => (
            <button
              key={scene.id}
              className={`scene-option ${selected === scene.id ? 'selected' : ''}`}
              onClick={() => setSelected(scene.id)}
              aria-pressed={selected === scene.id}
            >
              <span className="scene-option-icon">
                <Box size={20} />
              </span>
              <span>
                <strong title={scene.name}>{displayTitle(scene.name)}</strong>
                <small>{scene.description}</small>
                <em>
                  {scene.group}
                  {scene.id === currentScene ? ' · Current scene' : ''}
                </em>
              </span>
              {selected === scene.id ? <Check size={18} /> : <ArrowUpRight size={16} />}
            </button>
          ))}
          {!filtered.length && (
            <div className="empty-small">No matching scenes. Try another search.</div>
          )}
        </div>
      </div>
      <div className="scene-selection">
        <span className="eyebrow">{chosen ? 'Selected' : 'Driver scene catalog'}</span>
        <p>{chosen?.id ?? `${scenes.length} scenes available from the configured robot driver.`}</p>
      </div>
      {atCapacity && !switching && (
        <p className="warning-note">
          All 10 instance slots are occupied. Creating a scene will stop and delete the oldest open
          instance and its records.
        </p>
      )}
      <footer className="dialog-footer">
        <button className="button secondary" onClick={onClose}>
          Cancel
        </button>
        <button
          className="button primary"
          disabled={pending || !chosen || (switching && selected === currentScene)}
          onClick={() => onSelect(selected)}
        >
          {switching ? 'Confirm switch' : 'Create and start'}
          <ArrowUpRight size={16} />
        </button>
      </footer>
    </Dialog>
  )
}
