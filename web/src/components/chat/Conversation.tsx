import { useEffect, useRef, useState } from 'react'
import { ArrowDown, Check, Circle, LoaderCircle, Terminal, X } from 'lucide-react'
import type { Instance } from '../../api/types'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function Conversation({ instance }: { instance: Instance }) {
  const scroll = useRef<HTMLDivElement>(null)
  const follow = useRef(true)
  const [atBottom, setAtBottom] = useState(true)
  useEffect(() => {
    if (follow.current) scroll.current?.scrollTo({ top: scroll.current.scrollHeight })
  }, [instance.messages])

  return (
    <section className="conversation" aria-label="Conversation and execution">
      <header className="conversation-header">
        <span>Conversation</span>
        <span className="muted">
          {instance.messages.filter((message) => message.role === 'user').length} turns
        </span>
      </header>
      <div
        className="messages"
        ref={scroll}
        onScroll={() => {
          const element = scroll.current!
          follow.current = element.scrollHeight - element.scrollTop - element.clientHeight < 48
          setAtBottom(follow.current)
        }}
      >
        {instance.messages.length === 0 ? (
          <div className="chat-welcome">
            <span className="brand-mark">
              <img src="/emerge-logo.png" alt="" />
            </span>
            <h2>What should the robot do?</h2>
            <p>Send an instruction below. Follow the agent’s plan, actions, and results here.</p>
          </div>
        ) : (
          instance.messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              {message.role === 'user' && (
                <>
                  <span className="message-label">You</span>
                  <p>{message.text}</p>
                </>
              )}
              {message.role === 'assistant' && (
                <>
                  <span className="message-label">
                    <img src="/emerge-logo.png" alt="" />
                    EMERGE
                  </span>
                  <div className="markdown-message">
                    {message.text ? (
                      <Markdown remarkPlugins={[remarkGfm]}>{message.text}</Markdown>
                    ) : (
                      <span className="thinking">
                        Thinking<span>···</span>
                      </span>
                    )}
                  </div>
                </>
              )}
              {message.role === 'tool' && (
                <>
                  <div className="tool-heading">
                    {message.status === 'running' ? (
                      <LoaderCircle className="spin" size={15} />
                    ) : message.status === 'completed' ? (
                      <Check size={15} />
                    ) : (
                      <X size={15} />
                    )}
                    <span>{message.tool}</span>
                    <small>
                      {message.duration ??
                        (message.status === 'running'
                          ? 'Running'
                          : message.status === 'cancelled'
                            ? 'Stopped'
                            : 'Failed')}
                    </small>
                  </div>
                  <span className="tool-section-label">Arguments</span>
                  <pre>{message.arguments}</pre>
                  {message.text && <span className="tool-section-label">Result</span>}
                  <p className={message.status === 'failed' ? 'error-text' : ''}>{message.text}</p>
                  {message.verification && (
                    <div className="verification-result">
                      Verification: <strong>{message.verification.outcome}</strong>
                      {message.verification.predicates.map((predicate) => (
                        <p key={predicate.name}>
                          {predicate.name}:{' '}
                          {predicate.value === null ? 'uncertain' : String(predicate.value)} —{' '}
                          {predicate.evidence}
                        </p>
                      ))}
                    </div>
                  )}
                </>
              )}
              {message.role === 'system' && (
                <p className="system-message">
                  <Terminal size={14} />
                  {message.text}
                </p>
              )}
            </article>
          ))
        )}
        {instance.snapshot.plan.main_line.length > 0 && (
          <div className="run-plan" aria-label="Task steps">
            <span className="tool-section-label">Task plan</span>
            <p className="plan-mission">{instance.snapshot.plan.mission}</p>
            {instance.snapshot.plan.main_line.map((step) => (
              <div
                key={step.id}
                className={step.id === instance.snapshot.plan.pointer ? 'current' : ''}
              >
                {step.status === 'done' ? (
                  <Check size={13} />
                ) : step.status === 'active' && instance.runStatus === 'running' ? (
                  <LoaderCircle size={13} className="spin" />
                ) : (
                  <Circle size={12} />
                )}
                <span title={step.done_criterion}>
                  {step.subgoal}
                  {step.retries > 0 && <small> · {step.retries} retries</small>}
                </span>
              </div>
            ))}
            {instance.snapshot.plan.branch_stack?.map((branch) => (
              <p className="muted" key={branch.id}>
                Recovery: {branch.subgoal}
                <br />
                {branch.done_criterion}
              </p>
            ))}
          </div>
        )}
      </div>
      {!atBottom && (
        <button
          className="back-latest"
          onClick={() => {
            follow.current = true
            setAtBottom(true)
            scroll.current?.scrollTo({ top: scroll.current.scrollHeight, behavior: 'smooth' })
          }}
        >
          <ArrowDown size={15} />
          Jump to latest
        </button>
      )}
    </section>
  )
}
