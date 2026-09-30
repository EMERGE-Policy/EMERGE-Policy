import { useState } from 'react'
import { ArrowLeft, ArrowRight, Check, LoaderCircle } from 'lucide-react'
import { api } from '../../api/client'
import type { SetupStatus } from '../../api/types'

export default function SetupWizard({
  status,
  onComplete,
}: {
  status: SetupStatus
  onComplete: (status: SetupStatus) => void
}) {
  const [step, setStep] = useState(1)
  const [provider, setProvider] = useState(
    () => status.providers.find((item) => item.name === status.provider)!,
  )
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(provider.fields.map((field) => [field.name, field.default])),
  )
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')

  return (
    <main className="setup-page">
      <section className="setup-card" aria-labelledby="setup-title">
        <header className="setup-brand">
          <img src="/emerge-logo.png" alt="" />
          <span>EMERGE</span>
          <span className="eyebrow">WORKSPACE / FIRST RUN</span>
        </header>
        <ol className="setup-steps" aria-label="Setup progress">
          <li aria-current={step === 1 ? 'step' : undefined}>
            <span>{step === 2 ? <Check size={14} /> : '1'}</span> Provider
          </li>
          <li aria-current={step === 2 ? 'step' : undefined}>
            <span>2</span> Connection
          </li>
        </ol>
        <h1 id="setup-title">{step === 1 ? 'Connect your model.' : `Set up ${provider.label}.`}</h1>
        <p className="setup-description">
          {step === 1
            ? 'Choose the model service for your robot workspace.'
            : 'Add the connection details to get started.'}
        </p>
        <form
          onSubmit={async (event) => {
            event.preventDefault()
            if (step === 1) {
              setStep(2)
              return
            }
            setPending(true)
            setError('')
            try {
              onComplete(
                await api<SetupStatus>('/setup', 'POST', { provider: provider.name, ...values }),
              )
            } catch (error) {
              setError(String(error))
            } finally {
              setPending(false)
            }
          }}
        >
          {step === 1 ? (
            <fieldset className="setup-providers">
              <legend className="sr-only">Model provider</legend>
              {status.providers.map((item) => (
                <label
                  key={item.name}
                  className={`setup-provider ${provider.name === item.name ? 'selected' : ''}`}
                >
                  <input
                    type="radio"
                    name="provider"
                    value={item.name}
                    checked={provider.name === item.name}
                    onChange={() => {
                      setProvider(item)
                      setValues(
                        Object.fromEntries(item.fields.map((field) => [field.name, field.default])),
                      )
                      setError('')
                    }}
                  />
                  <span>
                    <strong>{item.label}</strong>
                    <small>{item.category}</small>
                  </span>
                </label>
              ))}
            </fieldset>
          ) : provider.isOauth ? (
            <p className="setup-oauth">
              This provider uses browser sign-in through the terminal setup. Run <code>emerge</code>{' '}
              on the server with the same <code>--config</code> option, complete sign-in, then
              restart the Web service and reload this page. You can also go back and choose an API
              provider.
            </p>
          ) : (
            <fieldset className="setup-fields" disabled={pending}>
              <legend className="sr-only">Connection details</legend>
              {provider.fields.map((field, index) => (
                <div key={field.name}>
                  <label className="form-label" htmlFor={`setup-${field.name}`}>
                    {field.title}
                    {!field.required && <span className="muted"> · Optional</span>}
                    <input
                      id={`setup-${field.name}`}
                      type={field.password ? 'password' : 'text'}
                      autoComplete={field.password ? 'new-password' : 'off'}
                      autoCapitalize="none"
                      spellCheck={false}
                      autoFocus={index === 0}
                      required={field.required}
                      aria-describedby={`setup-${field.name}-help`}
                      value={values[field.name]}
                      onChange={(event) =>
                        setValues((current) => ({ ...current, [field.name]: event.target.value }))
                      }
                    />
                  </label>
                  <p className="setup-field-help" id={`setup-${field.name}-help`}>
                    {field.description}
                  </p>
                </div>
              ))}
            </fieldset>
          )}
          {error && (
            <p className="setup-error error-text" role="alert">
              {error}
            </p>
          )}
          <footer className="setup-footer">
            {step === 2 ? (
              <button
                type="button"
                className="button secondary"
                disabled={pending}
                onClick={() => {
                  setStep(1)
                  setError('')
                }}
              >
                <ArrowLeft size={16} />
                Back
              </button>
            ) : (
              <span className="muted">Step 1 of 2</span>
            )}
            {!(step === 2 && provider.isOauth) && (
              <button
                className="button primary"
                disabled={
                  pending ||
                  (step === 2 &&
                    provider.fields.some((field) => field.required && !values[field.name].trim()))
                }
              >
                {pending ? <LoaderCircle className="spin" size={16} /> : null}
                {step === 1 ? 'Continue' : pending ? 'Saving…' : 'Save and enter workspace'}
                {!pending && <ArrowRight size={16} />}
              </button>
            )}
          </footer>
        </form>
        <p className="setup-note">
          Saved in your server’s Emerge configuration. API keys are not stored in this browser.
        </p>
      </section>
    </main>
  )
}
