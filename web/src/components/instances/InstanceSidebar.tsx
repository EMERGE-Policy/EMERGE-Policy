import { useState } from 'react'
import {
  Archive,
  Box,
  Check,
  MoreHorizontal,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Plus,
  Power,
  Search,
  Settings,
  Trash2,
} from 'lucide-react'
import type { Instance, Workspace } from '../../api/types'
import { statusLabels } from '../../api/types'
import { displayTitle } from '../../presentation'

export type InstanceCommand = 'rename' | 'close' | 'delete' | 'archive' | 'restore'

export default function InstanceSidebar({
  state,
  onSelect,
  onNew,
  onCollapse,
  onSettings,
  onCommand,
}: {
  state: Workspace
  onSelect: (id: string) => void
  onNew: () => void
  onCollapse: () => void
  onSettings: () => void
  onCommand: (item: Instance, command: InstanceCommand) => void
}) {
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('all')
  const today = new Date().toDateString()
  const filtered = state.instances.filter(
    (item) =>
      item.title.toLowerCase().includes(query.toLowerCase()) &&
      (filter === 'archived' ? item.archived : !item.archived) &&
      (filter !== 'active' || item.status !== 'closed'),
  )
  return (
    <aside className={`sidebar ${state.collapsed ? 'collapsed' : ''}`} aria-label="Instances">
      <div className="sidebar-brand">
        <img src="/emerge-logo.png" alt="EMERGE" />
        {!state.collapsed && <span>EMERGE</span>}
        <button
          className="icon-button"
          onClick={onCollapse}
          title={state.collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-label={state.collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {state.collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
        </button>
      </div>
      <button
        className="new-instance"
        title="New conversation"
        aria-label="New conversation"
        onClick={onNew}
        disabled={!state.connected}
      >
        <Plus size={19} />
        {!state.collapsed && (
          <>
            <span>New conversation</span>
            <kbd aria-hidden="true">＋</kbd>
          </>
        )}
      </button>
      {!state.collapsed && (
        <>
          <label className="search-field">
            <Search size={16} />
            <input
              placeholder="Search instances"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <div className="sidebar-section">
            <span>Your workspace</span>
            <select
              aria-label="Filter instances"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
            >
              <option value="all">All</option>
              <option value="active">Open</option>
              <option value="archived">Archived</option>
            </select>
          </div>
          <div className="instance-list">
            {['Today', 'Earlier'].map((group) => {
              const items = filtered.filter(
                (item) =>
                  (new Date(item.createdAt).toDateString() === today ? 'Today' : 'Earlier') ===
                  group,
              )
              return (
                items.length > 0 && (
                  <section key={group}>
                    <h3>{group}</h3>
                    {items.map((item) => (
                      <div
                        key={item.id}
                        className={`instance-row ${state.selectedId === item.id ? 'active' : ''}`}
                      >
                        <button className="instance-select" onClick={() => onSelect(item.id)}>
                          <Box size={17} />
                          <span>
                            <strong title={item.title}>{displayTitle(item.title)}</strong>
                            <small>
                              <i
                                className={`status-dot ${item.status === 'closed' ? 'off' : item.status === 'error' ? 'failed' : ''}`}
                              />
                              {item.runStatus === 'running'
                                ? 'Task running'
                                : statusLabels[item.status]}
                            </small>
                          </span>
                        </button>
                        <details className="menu instance-menu">
                          <summary className="icon-button" aria-label={`Actions for ${item.title}`}>
                            <MoreHorizontal size={17} />
                          </summary>
                          <div
                            className="menu-popover"
                            onClick={(event) =>
                              event.currentTarget.closest('details')?.removeAttribute('open')
                            }
                          >
                            <button onClick={() => onCommand(item, 'rename')}>
                              <Pencil size={15} />
                              Rename
                            </button>
                            <button
                              disabled={!!item.operation || !state.connected}
                              onClick={() => onCommand(item, item.archived ? 'restore' : 'archive')}
                            >
                              <Archive size={15} />
                              {item.archived ? 'Unarchive' : 'Archive'}
                            </button>
                            <button
                              disabled={
                                item.status === 'closed' || !!item.operation || !state.connected
                              }
                              onClick={() => onCommand(item, 'close')}
                            >
                              <Power size={15} />
                              Close scene
                            </button>
                            <hr />
                            <button
                              className="danger"
                              disabled={!!item.operation || !state.connected}
                              onClick={() => onCommand(item, 'delete')}
                            >
                              <Trash2 size={15} />
                              Delete
                            </button>
                          </div>
                        </details>
                      </div>
                    ))}
                  </section>
                )
              )
            })}
            {!filtered.length && <p className="empty-small">No matching instances</p>}
          </div>
          <div className="sidebar-note">
            <Check size={13} />
            <span>Isolated scenes & workspaces</span>
          </div>
        </>
      )}
      <footer className="sidebar-footer">
        <button onClick={onSettings} title="Settings" aria-label="Settings">
          <Settings size={18} />
          {!state.collapsed && <span>Settings</span>}
        </button>
        {!state.collapsed && (
          <span className="connection-badge">{state.connected ? 'Connected' : 'Reconnecting'}</span>
        )}
      </footer>
    </aside>
  )
}
