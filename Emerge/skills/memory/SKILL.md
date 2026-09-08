---
name: memory
description: Persistent long-term memory for important facts that should remain available across sessions.
metadata: {"Emerge":{"always":true}}
---

# Memory

`memory/MEMORY.md` stores long-term facts such as preferences, project context, relationships, and important notes. It is loaded into every conversation.

## When to Update MEMORY.md

Write important facts immediately using `edit_file` or `write_file`:

- User preferences ("I prefer dark mode")
- Project context ("The API uses OAuth2")
- Relationships ("Alice is the project lead")

## Auto-consolidation

Old conversations are automatically summarized into `MEMORY.md` when the session grows large. You do not need to manage this manually.
