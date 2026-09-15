# Emerge Package Documentation

`Emerge` is the project's main Agent package. It is responsible for organizing the LLM context, skills, tools, session state, embodied actions, and registered specialized subagents. The robot and simulation environment run from `robot/` at the repository root, and the two interact through workspace files.

## 1. Core Architecture

```text
User task
   ↓
AgentLoop
   ├── ContextBuilder：AGENTS.md, embodied state, memory, skills
   ├── ToolRegistry：files, Shell, planning, messaging, embodied actions, scene queries
   ├── SessionManager：session history
   └── delegate_subagent
          ↓
      SubagentRegistry
          ├── ObjectLocationSubagent
          ├── private context
          ├── object-localization skill
          ├── observe_scene
          ├── segment_candidates → VGGT + SAM3 + full-view overlays
          └── locate_candidates → reuse cached geometry/masks for verified candidates
          └── TaskVerificationSubagent
              ├── private context
              ├── object-state-verification skill
              ├── observe_scene
              └── submit_task_verification → submit visible state and evidence
```

The main Agent does not directly operate a subagent's internal tools. Before an action, it obtains precise world coordinates through `object_location`; after an action, it uses `task_verification` to verify whether the target object's intended state has actually been achieved.

## 2. Module Map

```text
Emerge/
├── __main__.py                    # python -m Emerge
├── agent/
│   ├── loop.py                    # Main Agent reasoning and tool loop
│   ├── context.py                 # Main Agent system prompt
│   ├── memory.py                  # Long-term memory and context compression
│   ├── skills.py                  # Built-in and workspace skill loading
│   └── tools/
│       ├── delegate.py            # Invoke registered specialized subagents
│       ├── embodied.py            # Write robot actions
│       ├── filesystem.py          # read / write / edit / list
│       ├── message.py             # Current task progress messages
│       ├── scene_graph.py         # Query ROBOT_STATE.md
│       ├── shell.py               # workspace Shell tools
│       └── update_plan.py         # Update PLAN.md status
│
├── base/
│   ├── tool.py                    # Tool abstraction interface
│   └── registry.py                # ToolRegistry
│
├── bus/                           # Message queue used by the CLI and AgentLoop
├── cli/commands.py                # onboard / agent / status / provider
├── config/                        # Configuration schema, loading, and runtime paths
├── providers/                     # LiteLLM, Azure, Codex, and compatible endpoints
├── session/                       # workspace/sessions/*.jsonl
├── skills/                        # Built-in Main Agent skills
│   ├── object-location/           # When to delegate precise localization
│   └── task-verification/         # When to delegate post-action verification
├── templates/                     # Initial workspace templates
│
└── subagents/
    ├── base.py                    # Shared BaseSubagent execution loop
    ├── content.py                 # Multimodal input such as text and images
    ├── context.py                 # Independent context for each invocation
    ├── models.py                  # Tasks, descriptors, and arbitrary structured results
    ├── registry.py                # Registry of complete subagent instances
    ├── skills.py                  # Private subagent SkillRegistry
    └── object_location/
        ├── main.py                # Standalone conversation entry point
        ├── register.py            # Assemble and register capabilities within the instance
        ├── agent.py / context.py
        ├── skills/object-localization/SKILL.md
        └── tools/                 # Observation, candidate verification, and localization tools
    └── task_verification/
        ├── main.py                # Standalone conversation entry point
        ├── register.py            # Assemble the verification instance
        ├── agent.py / context.py
        ├── skills/object-state-verification/SKILL.md
        └── tools/                 # Multiview observation and structured result submission
```

## 3. Runtime Configuration

Default configuration file：

```text
~/.Emerge/config.json
```

Default workspace：

```text
~/.Emerge/workspace
```

The configuration for the two vision subagents is located under `subagents.objectLocation` and `subagents.taskVerification`:

```json
{
  "subagents": {
    "objectLocation": {
      "model": "openai/gpt-5.5",
      "vggtUrl": "ws://localhost:8001",
      "sam3Url": "ws://localhost:8002",
      "timeout": 120.0,
      "maxIterations": 8,
      "viewCenterToleranceM": 0.08,
      "rayConsensusToleranceM": 0.02
    },
    "taskVerification": {
      "model": "openai/gpt-5.5",
      "maxIterations": 4
    }
  }
}
```

Runtime configuration keys use camelCase；in Python, the corresponding fields are `config.subagents.object_location` and
`config.subagents.task_verification`.
`viewCenterToleranceM` is used for the fallback consistency check of the VGGT depth center; the more accurate calibrated-ray consensus uses the separate
`rayConsensusToleranceM`.

## 4. Common Commands

Install from the repository root：

```bash
pip install -e .
```

The project requires Python 3.10 or later. Create the configuration for the first time or refresh it：

```bash
python -m Emerge onboard
# Equivalent: emerge onboard
```

Start the Main Agent interactive terminal：

```bash
python -m Emerge agent
# Equivalent: emerge agent
```

Run a single task：

```bash
python -m Emerge agent -m "Find the salad dressing"
```

Specify the configuration, workspace, and session：

```bash
python -m Emerge agent \
  --config ~/.Emerge/config.json \
  --workspace ~/.Emerge/workspace \
  --session cli:direct
```

Check the current configuration status：

```bash
python -m Emerge status
```

Log in to an OAuth provider：

```bash
python -m Emerge provider login openai-codex
python -m Emerge provider login github-copilot
```

There are currently no `chat` or `gateway` subcommands; use `agent` as the unified interactive entry point.

## 5. External Model Services

OpenPI, VGGT, and SAM3 are all located in `external_model_server/` at the repository root. Use the following unified startup command：

```bash
bash scripts/model_server/start_external_model_servers.sh
```

The default assignments are as follows：

| Service | Port | conda environment |
|---|---:|---|
| OpenPI | 8000 | `pi05_server` |
| VGGT | 8001 | `EmergePolicy` |
| SAM3 | 8002 | `EmergePolicy` |

The GPU and OpenPI batch parameters can be overridden through environment variables：

```bash
OPENPI_GPU=3,4,5 \
OPENPI_MAX_BATCH_SIZE=1 \
VGGT_GPU=1 \
SAM3_GPU=2 \
bash scripts/model_server/start_external_model_servers.sh
```

## 6. Running the Object Location Subagent Independently

Interactive mode：

```bash
python -m Emerge.subagents.object_location.main
```

Single task：

```bash
python -m Emerge.subagents.object_location.main \
  --task "Find the salad dressing"
```

You can also use `--config`, `--workspace`, and `--model` to override runtime parameters. This instance requires the following file to already exist in the workspace：

```text
artifacts/observations/observation.json
```

Each enabled view in the manifest should provide an image path, `intrinsics`, and `T_world_camera`. The Controller saves each camera's image by overwriting it and updates the manifest by overwriting it as well. During localization, the entire manifest participates in VGGT reconstruction, after which consistent views are selected when merging the target point cloud.

The current localization data flow consistently uses the cameras' original `512×512` RGB images: the multimodal model views the original images, SAM3 performs segmentation on the originals, VGGT outputs depth and point maps at the same size, and mask overlays also remain at 512. Internally, VGGT pads only the right and bottom edges to 518 to satisfy the 14-pixel patch requirement. It crops the output back to 512 immediately after inference, without changing the intrinsics or pixel coordinates.

The subagent reads only the current complete set of views; it does not read object reference images or `ROBOT_STATE.md`. Based on the semantic target provided by the user and the visible evidence in the scene, the multimodal model compares all reasonable physical candidates and then generates a pure-appearance SAM3 prompt for each candidate. `segment_candidates` segments all candidates in one call and returns mask overlays containing `candidate_id` as multimodal results. The subagent reviews every view, submits the `verified_views` in which the same object is a true match, and then calls `locate_candidates`.

The final localization uses bounding-box ray consensus under known camera calibration to reject masks that jump to other objects. It uses the triangulated center to correct each view's local VGGT point cloud and then performs pose estimation. The final step reuses the VGGT geometry and SAM3 masks cached during the current run and does not request either model server again.

## 7. Running the Task Verification Subagent Independently

Interactive mode：

```bash
python -m Emerge.subagents.task_verification.main
```

Single verification：

```bash
python -m Emerge.subagents.task_verification.main \
  --task "Verify whether the apple is inside the basket and released"
```

This instance reads the same `artifacts/observations/observation.json` and the current complete set of views, decomposes the target outcome into visible conditions, and then submits evidence through `submit_task_verification`. It does not invoke VGGT, SAM3, or coordinate localization. The overall result is computed by code from all required conditions as `achieved`, `not_achieved`, or `uncertain`.

## 8. Main Agent Invocation of Subagents

The Main Agent has two corresponding resident skills: `object-location` guides precise localization before an action, and `task-verification` guides post-action result checking. Both invoke complete subagents through `delegate_subagent`, but they use the registered names `object_location` and `task_verification`, respectively.

The tool registered by the Main Agent is named `delegate_subagent`. The internal arguments issued by the model take the following form：

```json
{
  "agent_name": "object_location",
  "task": "Locate the salad dressing precisely and describe nearby obstacles."
}
```

This invocation is an asynchronous coroutine, but the current Main Agent turn `await`s the localization result. It does not block the entire asyncio event loop, nor is it a background task that returns immediately after startup.

The returned content includes：

```json
{
  "status": "success",
  "agent_name": "object_location",
  "summary": "Localized: salad_dressing.",
  "output": {
    "objects": [
      {
        "name": "salad_dressing",
        "found": true,
        "frame": "world",
        "position_m": [0.0997, -0.1923, 0.0503],
        "size_m": [0.0469, 0.0367, 0.1176],
        "rpy_rad": [0.0, 0.0, -1.10]
      }
    ],
    "scene_context": "The target is near several containers and a woven basket."
  },
  "error": null
}
```

Example of a post-action verification invocation：

```json
{
  "agent_name": "task_verification",
  "task": "Verify whether the apple is inside the basket and has been released by the gripper."
}
```

The corresponding condensed result takes the following form：

```json
{
  "status": "success",
  "agent_name": "task_verification",
  "summary": "Verification outcome: achieved.",
  "output": {
    "outcome": "achieved",
    "predicates": [
      {
        "name": "apple_inside_basket",
        "value": true,
        "evidence": "The apple is visibly below the basket rim."
      },
      {
        "name": "apple_released",
        "value": true,
        "evidence": "The open gripper is separated from the apple."
      }
    ],
    "scene_context": "The basket remains upright."
  },
  "error": null
}
```

The Main Agent receives only the condensed fields shown above. The SAM prompts, candidate IDs, view selection, ray residuals, point-cloud counts, and geometric alignment information from the localization process remain internal to `object_location`. Neither subagent directly controls the robot; the Main Agent continues planning and invokes embodied-action tools based on the localization or verification result.

When multiple related objects need to be localized during the same action phase, delegate them together in a single `task` so they share the same observations and reconstruction results. When `found: false` is returned, the Main Agent must treat the position as unknown and must not guess world coordinates. Localization and post-action verification should be performed in two separate delegations because they correspond to different moments in the physical scene. If the verification result is `not_achieved`, continue correcting the action; if it is `uncertain`, gather additional observations. Neither result can be treated as task success.

## 9. Workspace Contract

| File | Purpose |
|---|---|
| `AGENTS.md` | Main Agent instructions |
| `EMBODIED.md` | Description of current embodied-system capabilities |
| `ROBOT_STATE.md` | `Robot and scene runtime state written back by the Controller |
| `PLAN.md` | Current task status of the Main Agent |
| `ACTION.md` | Action queue consumed by the Controller |
| `memory/MEMORY.md` | Long-term memory |
| `artifacts/observations/observation.json` | Multicamera image and calibration manifest |
| `sessions/*.jsonl` | Session history |

## 10. Extension Points

- Add a Main Agent tool: implement `Tool` in `agent/tools/` and register it in `AgentLoop._register_default_tools()`.
- Add a Main Agent skill: create `skills/<name>/SKILL.md`.
- Add a specialized subagent: place its own agent, context, skills, tools, and `register.py` in `subagents/<name>/`; after assembly, register the instance with the Main Agent's `SubagentRegistry`.
- Add a provider: extend `providers/registry.py`, the provider implementation, and `config/schema.py`.
- Add a workspace contract: add a template and include it in `ContextBuilder`'s loading list as needed.

If code behavior and this documentation are inconsistent, the current code takes precedence.
