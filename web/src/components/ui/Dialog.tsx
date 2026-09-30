import { useEffect, useRef, type ReactNode } from 'react'
import { X } from 'lucide-react'

export default function Dialog({
  title,
  children,
  onClose,
  wide = false,
}: {
  title: string
  children: ReactNode
  onClose: () => void
  wide?: boolean
}) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const dialog = ref.current!
    const previous = document.activeElement as HTMLElement | null
    dialog.showModal()
    return () => {
      dialog.close()
      previous?.focus()
    }
  }, [])
  return (
    <dialog
      ref={ref}
      className={wide ? 'dialog wide' : 'dialog'}
      aria-labelledby="dialog-title"
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className="dialog-inner">
        <header className="dialog-header">
          <h2 id="dialog-title">{title}</h2>
          <button className="icon-button" aria-label="Close dialog" onClick={onClose}>
            <X size={19} />
          </button>
        </header>
        {children}
      </div>
    </dialog>
  )
}
