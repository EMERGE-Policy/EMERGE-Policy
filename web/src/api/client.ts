export async function api<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) {
    const text = await response.text()
    let detail: string
    try {
      detail = JSON.parse(text).detail
    } catch {
      detail = text || response.statusText
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return response.json()
}
