# Emerge

`Emerge` is the agent orchestration and execution package behind EMERGE-Policy. It turns a user instruction into a stateful embodied-agent run by combining an LLM, workspace context, specialist sub-agents, robot tools, memory, and a versioned runtime protocol.

The package does not own the simulator or robot process. Those components live under `robot/` and exchange state and actions with Emerge through a shared workspace.

## System overview

```text
Human TUI or machine client
            |
            v
      AgentRuntime
      - configuration loading
      - exclusive workspace lease
      - events and run artifacts
      - cancellation and cleanup
            |
            v
        AgentLoop
      - provider and sessions
      - context and memory
      - tools and skills
      - visual monitor
            |
            +-----------------------------+
            |                             |
            v                             v
  Specialist sub-agents          Workspace action queue
  - object_location              ACTION.md -> controller
  - task_verification            controller -> ROBOT_STATE.md
            |
            v
  Observation and perception services
  - calibrated camera views
  - VGGT reconstruction
  - SAM3 segmentation
```

The TUI and headless CLI are clients of the same `AgentRuntime`. They therefore use the same configuration, workspace lock, agent loop, event model, cancellation rules, and cleanup behavior.

## Main capabilities

- Stateful tool-using agent loop with context-window management and persistent sessions.
- Workspace-based robot coordination through explicit state, plan, observation, and action files.
- Geometry-driven motion primitives plus model-policy execution through either VLA or WAM.
- Dedicated object-localization and task-verification sub-agents with isolated contexts and tools.
- Continuous visual monitoring that can interrupt a running robot action when its visible subgoal is achieved.
- Interactive full-screen terminal UI for human operation.
- Versioned JSON request, event, and result contracts for evaluation and automation.
- Provider routing for hosted APIs, OAuth providers, OpenAI-compatible endpoints, and local deployments.

## Package map

```text
Emerge/
├── agent/
│   ├── loop.py                 # Main reasoning and tool-use loop
│   ├── context.py              # Workspace, memory, skill, and robot context
│   ├── memory.py               # Context compaction and long-term memory
│   ├── visual_monitor.py       # Asynchronous visual action monitor
│   └── tools/                  # Files, shell, plan, delegation, and robot actions
├── base/                       # Tool interfaces and registries
├── bus/                        # Runtime message queue
├── cli/
│   ├── commands.py             # `emerge` entry point
│   ├── management.py           # Workspace and provider commands
│   └── headless.py             # Machine-only single-run client
├── config/                     # Pydantic schema, loading, and paths
├── providers/                  # Provider registry and implementations
├── runtime/
│   ├── protocol.py             # RunRequest, RunEvent, and RunResult
│   ├── service.py              # Shared execution authority
│   ├── storage.py              # Workspace lease and run artifacts
│   └── snapshots.py            # Workspace and service status snapshots
├── session/                    # Persistent conversation sessions
├── skills/                     # Built-in main-agent skills
├── subagents/
│   ├── object_location/        # Multi-view object localization
│   └── task_verification/      # Visual post-action verification
├── templates/                  # Initial workspace documents
└── tui/                        # Full-screen interactive client
```

## Installation

Install the project from the repository root:

```bash
pip install -e .
```

Python 3.10 or later is required. Robot environments, external model servers, checkpoints, and evaluation dependencies are documented in the repository-level README and the guides under `scripts/`.

## Initialize the workspace

Create the default configuration and workspace:

```bash
emerge workspace init
```

This creates:

```text
~/.Emerge/config.json
~/.Emerge/workspace/
```

Running the command again refreshes the configuration schema and adds missing workspace templates without replacing existing workspace files unless configuration overwrite is explicitly confirmed.

Inspect the active paths, model, and provider state with:

```bash
emerge workspace status
```

## Configuration

Configuration files use camelCase keys. A compact configuration looks like this:

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/.Emerge/workspace",
      "model": "anthropic/claude-opus-4-5",
      "provider": "auto",
      "maxTokens": 8192,
      "contextWindowTokens": 65536,
      "temperature": 0.1,
      "maxToolIterations": 40
    }
  },
  "visualMonitor": {
    "enabled": true,
    "pollIntervalSeconds": 0.5,
    "verificationTimeoutSeconds": 30.0,
    "confirmations": 1
  },
  "subagents": {
    "objectLocation": {
      "model": null,
      "vggtUrl": "ws://localhost:8001",
      "sam3Url": "ws://localhost:8002",
      "timeout": 120.0,
      "maxIterations": 8
    },
    "taskVerification": {
      "model": null,
      "maxIterations": 2
    }
  },
  "providers": {
    "anthropic": {
      "apiKey": "YOUR_API_KEY"
    }
  },
  "tools": {
    "restrictToWorkspace": false,
    "exec": {
      "timeout": 60,
      "pathAppend": ""
    }
  }
}
```

When a specialist sub-agent has no model override, it uses the main agent model. Set `tools.restrictToWorkspace` to `true` when file and shell access must remain inside the configured workspace.

The provider registry supports Anthropic, OpenAI, OpenRouter, Azure OpenAI, DeepSeek, Gemini, Groq, DashScope, Moonshot, MiniMax, Zhipu, AiHubMix, SiliconFlow, VolcEngine, custom OpenAI-compatible endpoints, vLLM, Ollama, OpenAI Codex OAuth, and GitHub Copilot OAuth. With `provider` set to `auto`, Emerge resolves the provider from the model prefix and configured credentials.

OAuth login commands are:

```bash
emerge provider login openai-codex
emerge provider login github-copilot
```

## Interactive client

Start the full-screen terminal client:

```bash
emerge
```

Optional overrides apply to the runs started by that client:

```bash
emerge \
  --config ~/.Emerge/config.json \
  --workspace ~/.Emerge/workspace \
  --session cli:demo \
  --model anthropic/claude-opus-4-5
```

Type `/` in an empty prompt to open the command palette. Available operations include starting a new session, resuming a saved session, changing the model, stopping a run, checking service health, viewing logs, toggling tool details, inspecting workspace paths, exporting the conversation, and viewing recent run artifacts.

## Headless runtime

Use the headless client for evaluations, scripts, and other machine integrations:

```bash
python -m Emerge.cli.headless \
  "Put the red block in the basket" \
  --workspace ~/.Emerge/workspace \
  --session eval:episode-001 \
  --timeout-s 600
```

It writes exactly one `RunResult` JSON object to standard output. Diagnostics go to standard error. Unless `--output-dir` is provided, artifacts are stored under:

```text
<workspace>/runs/<run_id>/
├── request.json
├── events.jsonl
├── result.json
└── runtime.log
```

A request file can provide the versioned contract directly:

```json
{
  "schema_version": "Emerge.run_request.v1",
  "run_id": "episode_001",
  "session_id": "eval:episode-001",
  "message": "Put the red block in the basket",
  "workspace": "/absolute/path/to/workspace",
  "timeout_s": 600,
  "cancel_timeout_s": 10,
  "stream": true,
  "metadata": {
    "suite": "libero"
  }
}
```

Run it with:

```bash
python -m Emerge.cli.headless --request request.json
```

`RunResult.run_status` is one of `completed`, `failed`, `cancelled`, or `timed_out`. Cancellation data acknowledges whether physical execution was stopped; it is not a benchmark success signal.

Only one runtime may own a workspace at a time. `WorkspaceLease` prevents concurrent writers from issuing conflicting robot actions or mutating the same run state.

## Agent context and workspace contract

The workspace is the boundary between language-level reasoning and the robot controller. The runtime creates missing templates before a run, while preserving existing files.

| Path | Owner and purpose |
|---|---|
| `AGENTS.md` | User-maintained instructions for the main agent |
| `EMBODIED.md` | Active robot capabilities and action conventions |
| `ROBOT_STATE.md` | Controller-written robot and scene state |
| `PLAN.md` | Agent-maintained task plan and progress |
| `ACTION.md` | Action queue shared by the agent and controller |
| `memory/MEMORY.md` | Consolidated long-term memory |
| `artifacts/observations/observation.json` | Current multi-camera observation manifest |
| `sessions/*.jsonl` | Persistent conversation histories |
| `runs/<run_id>/` | Structured headless run artifacts |
| `skills/` | Optional workspace-local skills |

At the start of a turn, the context builder combines the active instructions, robot state, plan, memory, and relevant skills. The agent must re-read `ROBOT_STATE.md` after each physical action because the copy included in its original context becomes stale as soon as the controller moves the robot.

## Tools and skills

The main agent registers these tools:

| Tool | Purpose |
|---|---|
| `read_file`, `write_file`, `edit_file`, `list_dir` | Inspect and update workspace files |
| `exec` | Run a shell command with configured timeout and path policy |
| `update_plan` | Update protected plan state safely |
| `message` | Emit a concise progress message |
| `delegate_subagent` | Run a registered specialist sub-agent |
| `execute_robot_action` | Validate, enqueue, monitor, and await a robot action |
| `query_scene_graph` | Query structured facts from current robot state |

Built-in skills define operating procedures for planning, progress reporting, memory, object localization, task verification, geometric attachment, VLA control, and WAM control. Workspace-local skills can be added under `<workspace>/skills/<name>/SKILL.md`.

### Robot action backends

`execute_robot_action` always exposes geometry-driven actions such as `move_to_pose`, `move_linear`, `set_gripper`, and `follow_arc`. A run may additionally enable one model-policy backend:

```bash
EMERGE_POLICY_BACKEND=vla   # enables vla_execute
EMERGE_POLICY_BACKEND=wam   # enables wam_execute
```

The evaluation launcher normally sets this variable. When a backend is selected, the other model-policy action is rejected, while geometry-driven actions remain available.

For VLA, the agent provides a phase-local natural-language instruction and a step budget. For WAM, the agent provides the next unfinished contact phase; the evaluator separately supplies and locks the task-conditioning mode and full task instruction.

Each accepted action is appended to `ACTION.md`. The controller executes it, updates its status, and refreshes `ROBOT_STATE.md`. The tool waits for a terminal action result and participates in runtime cancellation, so a run cannot report clean completion while robot actions remain active.

## Specialist sub-agents

### Object location

The `object_location` sub-agent locates movable objects, destinations, obstacles, and visible task-relevant fixtures from calibrated camera views. Its private tool chain is:

```text
observe_scene
      -> segment_candidates (SAM3 + cached VGGT geometry)
      -> visual candidate verification
      -> locate_candidates (multi-view world-frame result)
```

It reads `artifacts/observations/observation.json`, compares plausible candidates across the enabled views, segments them, rejects inconsistent masks using calibrated-view consensus, and returns compact world-frame positions, dimensions, orientations, and scene context. Internal prompts, masks, residuals, and point-cloud diagnostics remain private to the sub-agent.

Run it directly for diagnosis:

```bash
python -m Emerge.subagents.object_location.main \
  --task "Locate the salad dressing"
```

### Task verification

The `task_verification` sub-agent inspects current multi-camera images after an action. It decomposes the requested result into visible predicates and submits an `achieved`, `not_achieved`, or `uncertain` result with evidence. It does not control the robot and does not require VGGT or SAM3.

Run it directly for diagnosis:

```bash
python -m Emerge.subagents.task_verification.main \
  --task "Verify that the apple is inside the basket and released"
```

### Visual monitor

When enabled, the visual monitor observes action progress independently of the main reasoning loop. It uses the current plan, action, and camera observations to detect visible subgoal completion. A confirmed result can request an early stop for the active action, after which the main agent resumes from freshly written robot state.

## Sessions and memory

Each run has a `session_id`. Messages are persisted in the workspace so the TUI or a later headless request can resume the same conversation. When the prompt approaches `contextWindowTokens`, the memory subsystem consolidates older information into long-term memory instead of relying on an unbounded message history.

Run token usage in `RunResult` currently covers the main agent. Specialist sub-agent usage is not included in that field.

## External services

Emerge connects to external services but does not launch them from inside the package:

- The robot or simulator controller consumes `ACTION.md`, writes results, updates `ROBOT_STATE.md`, and publishes observations.
- The object-location sub-agent connects to VGGT at `ws://localhost:8001` and SAM3 at `ws://localhost:8002` by default.
- VLA and WAM inference are executed by the active controller integration and their model servers.

Use the repository scripts under `scripts/model_server/` and the evaluation guides under `scripts/` to start the required service combination for a task.

## Extension points

- **Main-agent tool:** implement `Tool` under `agent/tools/` and register it in `AgentLoop._register_default_tools()`.
- **Main-agent skill:** add `skills/<name>/SKILL.md`, or install it only for one workspace under `<workspace>/skills/`.
- **Specialist sub-agent:** provide an isolated agent, context builder, skill registry, tool registry, and `register.py`, then add the assembled instance to the main `SubagentRegistry`.
- **Provider:** add a `ProviderSpec`, a matching configuration field, and a provider implementation when the LiteLLM adapter is insufficient.
- **Runtime client:** construct `RunRequest`, call `AgentRuntime.run()`, and consume `RunEvent` callbacks and the final `RunResult`.
- **Workspace protocol:** add a template and explicitly load or consume it in the context builder, agent tool, or controller.

## Development checks

Run the package tests from the repository root:

```bash
pytest tests
```

Useful interface checks:

```bash
emerge --help
emerge workspace --help
python -m Emerge.cli.headless --help
python -m Emerge.subagents.object_location.main --help
python -m Emerge.subagents.task_verification.main --help
```

When this document and the implementation disagree, the current code is authoritative.
