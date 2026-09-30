# EMERGE Web

The browser workspace uses the existing Emerge AgentRuntime and robot Controller. Each conversation owns a dedicated Controller process and workspace. Camera images come from that workspace's `artifacts/observations/observation.json` and its referenced files.

## Start the application

Run from the repository root. Complete the main project's LIBERO installation first.

```bash
conda activate EmergePolicy
python -m pip install -e '.[web]'
npm --prefix web install
npm --prefix web run build
emerge web
```

Open **http://127.0.0.1:8080/**. For a remote server, forward port 8080 over SSH. Node.js 22.12+ is needed for the frontend build; Python and Controller processes use the active `EmergePolicy` environment.

```bash
emerge web \
  --config ~/.Emerge/config.json \
  --driver libero_mujoco \
  --driver-config dev/libero_mujoco_driver_sample.json \
  --data-dir ~/.Emerge/web \
  --port 8080
```

`python -m Emerge.web.app` provides the same entry point. Run one Web service per data directory. The server holds an exclusive directory lease. Static files are served from `web/dist` by default; `--frontend` overrides that directory.

On first launch, the browser guides you through choosing a provider and entering the model, API address, and API key. It uses the same configuration checks, field rules, and save logic as the TUI. A missing configuration file is created only after completing setup; an existing, ready configuration skips the guide. Configuration is saved to `--config` or `~/.Emerge/config.json`, and takes effect without restarting the Web service. Existing keys can be kept by leaving the key field blank; keys are never returned to the page or saved in browser storage. Setup checks required fields, not remote credentials or model availability.

OAuth sign-in remains in the terminal setup: run `emerge` with the same `--config`, complete sign-in, then restart the Web service. The browser guide supports API providers and local endpoints.

Model servers are started separately with the existing repository scripts. Use the same policy backend and driver profile as the TUI setup. The default sample driver uses the LIBERO profile from `robot/profiles/libero_mujoco.md`. The Web service does not launch or stop shared model servers.

## Behavior

- New conversation reads the driver's real catalog, allocates an isolated workspace, and starts `robot.controller` with an instance-specific configuration. The initial image and Controller readiness gate sending.
- Instructions run through `Emerge.cli.headless` and the existing AgentRuntime. Each run receives its instance workspace, current session, model, and configuration. Child processes isolate per-run provider and environment state.
- Assistant replies and expanded tool details come from `RunEvent` records. Plans, actions, robot state, and observations come from existing workspace snapshots. Goal verification, environment success, and execution completion are shown separately.
- Server-sent events deliver authoritative state every 0.5 seconds. Browser reconnects replace the current projection using stable message IDs, without resubmitting instructions. Background work continues when a browser disconnects.
- Stop uses the headless cancellation signal and waits for robot-action acknowledgement. Reset/switch use the existing Controller handshake and task-workspace cleanup, starting a new session as the TUI does.
- Close releases the Controller and retains records. Delete reclaims processes before removing the instance directory. At 10 open instances, a new startup first deletes the oldest open instance. If that instance is changing environments or cannot stop/delete, the new startup reports an error without evicting another instance.
- The image selector lists only published cameras. The visible, active camera receives the same PNG observations over a WebSocket, independently of the 0.5-second workspace updates. After displaying a frame, the browser requests the newest available image; unchanged images are not resent and slow clients do not queue old frames. Background tabs pause the stream, and interrupted connections reconnect. Screenshots and closed scenes use the original PNG endpoint. Actual frame rate follows camera output, decoding, and network speed.
- Camera PNG encoding and writes use up to four threads per observation batch. Rendering and robot control stay on their existing thread, and the observation manifest is published only after the whole batch finishes. Resolution, pixel values, calibration, and control frequency are unchanged.
- Model selection starts with the configured model and previously selected instance models; the field also accepts a model identifier, matching the TUI `/model` behavior. Settings can probe shared model services. First-run API credentials are saved in the existing Python configuration.
- Logs, run artifacts, screenshots, existing recordings, and conversation export are available from the header's document button.
- English interface, system light/dark theme, collapsible sidebar, resizable panels, and browser-local per-instance drafts and layout preferences.

Runtime data lives under `~/.Emerge/web/instances/<id>/`. The existing `~/.Emerge/workspace` used by the TUI is separate. Normal Web service shutdown reclaims its children. On restart, persisted child identities are checked and leftover owned processes are reclaimed; affected scenes require explicit restart. Neither Controllers nor old instructions are automatically replayed.

## Frontend development

Keep the Python service running, then use:

```bash
npm --prefix web run dev
```

Vite serves the page at http://127.0.0.1:5173 and proxies `/api` to port 8080. Set `EMERGE_API_URL` when starting Vite to use another backend address. No mock transport or illustrative scene is used.
