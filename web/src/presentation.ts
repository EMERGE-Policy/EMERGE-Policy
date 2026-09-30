// LIBERO prefixes identify source scenes; keep the task itself readable in titles.
// Full identifiers remain available in tooltips and the scene catalog.
export function displayTitle(title: string) {
  const task = title.replace(/^[A-Z][A-Z_ ]*SCENE\s*\d+\s+/u, '')
  return task.charAt(0).toUpperCase() + task.slice(1)
}
