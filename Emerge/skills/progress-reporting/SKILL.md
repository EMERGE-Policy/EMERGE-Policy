---
name: progress-reporting
description: Report concise user-visible milestones with the message tool during long-running embodied tasks, sub-agent waits, recovery, or important plan changes without exposing private reasoning or replacing the final answer.
metadata: {"Emerge":{"always":true}}
---

# Progress Reporting

Use `message` only for a meaningful milestone in work that spans multiple tools or waits.

Report when:

- a specialist sub-agent is dispatched or its result changes the next step;
- a major plan phase completes and the next phase begins;
- recovery or replanning becomes necessary;
- an external operation is taking long enough that silence would be confusing.

Do not report greetings, ordinary file reads, every tool call, or facts that the automatic tool hint already shows. Do not expose hidden reasoning, chain-of-thought, credentials, or raw multimodal data. Do not use `message` as the final answer.

Keep each update to one or two factual sentences stating what changed and what happens next. When calling `message`, do not duplicate the same update in assistant text. Continue the task after reporting; the normal final response is always delivered separately.
