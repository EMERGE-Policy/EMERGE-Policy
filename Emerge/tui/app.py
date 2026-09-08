"""Full-screen human client. Execution remains in Emerge.runtime."""
from __future__ import annotations

import asyncio
import io
import json
import time
from collections import deque
from pathlib import Path
from uuid import uuid4

from loguru import logger
from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    ConditionalContainer, Float, FloatContainer, HSplit, Layout, VSplit, Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.widgets import Frame, TextArea
from rich.console import Console
from rich.markdown import Markdown

from Emerge.runtime.configuration import load_runtime_config
from Emerge.runtime.protocol import RunRequest
from Emerge.runtime.service import AgentRuntime
from Emerge.runtime.snapshots import service_health, workspace_snapshot
from Emerge.runtime.storage import atomic_json
from Emerge.runtime.workspace import reset_workspace_context
from Emerge.session.manager import SessionManager
from Emerge.tui.state import ViewState
from Emerge.tui.theme import style

BRAND = (
    "█▀▀ █▄█ █▀▀ █▀█ █▀▀ █▀▀  █▀█ █▀█ █   █ █▀▀ █ █",
    "█▀▀ █ █ █▀▀ ██▀ █▄█ █▀▀  █▀▀ █ █ █   █ █   ▀█▀",
    "▀▀▀ ▀ ▀ ▀▀▀ ▀ ▀ ▀▀▀ ▀▀▀  ▀   ▀▀▀ ▀▀▀ ▀ ▀▀▀  ▀ ",
)

COMMANDS = [
    ("/new", "Clear workspace and start a new session"),
    ("/sessions", "Search and resume sessions"),
    ("/model", "Set model for the next run"),
    ("/stop", "Stop the current run"),
    ("/refresh", "Refresh workspace data now"),
    ("/health", "Probe perception services"),
    ("/sidebar", "Show or hide workspace sidebar"),
    ("/details", "Show or hide tool details"),
    ("/logs", "Switch between conversation and logs"),
    ("/theme", "Switch dark or light theme"),
    ("/export", "Export visible conversation"),
    ("/workspace", "Show workspace path"),
    ("/observations", "Show observation manifest path"),
    ("/runs", "Show latest run artifacts"),
    ("/keys", "Show input controls"),
    ("/commands", "Open this command menu"),
    ("/quit", "Quit Emerge"),
]


class TranscriptLexer(Lexer):
    def lex_document(self, document):
        def line(index):
            text = document.lines[index]
            token = (
                "user" if text.startswith("YOU")
                else "assistant" if text.startswith("EMERGE")
                else "tool" if text.startswith("  ↳")
                else "system" if text.startswith("·")
                else ""
            )
            return [("class:" + token, text)]
        return line


class WorkspaceApp:
    SNAPSHOT_INTERVAL_S = 0.5
    SIDEBAR_WIDTH = 44

    def __init__(self, *, config=None, workspace=None, session_id=None, model=None, runtime=None):
        self.config_path = config
        self.config = load_runtime_config(RunRequest(
            message="UI configuration", config=config, workspace=workspace, model=model,
        ))
        self.workspace = self.config.workspace_path.expanduser().resolve()
        self.sessions = SessionManager(self.workspace)
        self.state = ViewState(
            session_id or "cli:" + uuid4().hex,
            self.config.agents.defaults.model,
        )
        self.runtime = runtime or AgentRuntime()
        self.cancel = None
        self.task = None
        self.closing = False
        self.details = False
        self.logs = deque(maxlen=300)
        self.show_logs = False
        self.last_output = None
        self.health = []
        self.dirty = True
        self.snapshot = None
        self.snapshot_time = 0.0
        self.sidebar_fragments = []
        self._entry_cache = {}
        self.preferences_path = self.workspace / ".tui.json"
        preferences = (
            json.loads(self.preferences_path.read_text())
            if self.preferences_path.exists() else {}
        )
        self.sidebar = preferences.get("sidebar", True)
        self.light = preferences.get("light", False)
        self.palette_open = False
        self.palette_entries = []
        self.palette_index = 0

        self.transcript = TextArea(
            read_only=True, scrollbar=True, wrap_lines=True, lexer=TranscriptLexer(),
        )
        self.editor = TextArea(
            height=4, multiline=True, style="class:input", prompt=" › ",
        )
        self.palette_input = TextArea(height=1, multiline=False, style="class:palette")
        self.palette_input.buffer.on_text_changed += self._palette_changed
        palette = Frame(HSplit([
            self.palette_input,
            Window(FormattedTextControl(self._palette_text), height=14),
        ]), title="Slash commands")
        self.side = Window(
            FormattedTextControl(lambda: self.sidebar_fragments),
            width=self.SIDEBAR_WIDTH, wrap_lines=True, always_hide_cursor=True,
        )
        body = VSplit([
            Frame(self.transcript, title="Conversation"),
            ConditionalContainer(
                Frame(self.side, title="Workspace"),
                filter=Condition(
                    lambda: self.sidebar
                    and self.application.output.get_size().columns >= 100
                ),
            ),
        ])
        root = FloatContainer(HSplit([
            Window(FormattedTextControl(self._header), height=4, style="class:header"),
            body,
            Frame(self.editor, title="Message · Enter send · / commands"),
            Window(FormattedTextControl(self._footer), height=1, style="class:footer"),
        ]), floats=[
            Float(
                content=ConditionalContainer(
                    palette, filter=Condition(lambda: self.palette_open),
                ),
                width=72, top=4,
            ),
        ])
        self.application = Application(
            layout=Layout(root, focused_element=self.editor),
            key_bindings=self._bindings(),
            style=style(self.light),
            full_screen=True,
            mouse_support=True,
            min_redraw_interval=0.03,
        )
        self._restore_session()
        if model:
            self.state.model = model
        self.refresh(force_snapshot=True)

    @property
    def busy(self):
        return self.task is not None and not self.task.done()

    def _header(self):
        columns = self.application.output.get_size().columns
        if columns < 64:
            return [
                ("class:brand.name", "  EmergePolicy\n"),
                ("class:brand.meta", f"  {self.workspace.name}  ·  {self.state.status}\n"),
                ("class:brand.meta", f"  {self.state.model}"),
            ]
        logo = "\n".join("  " + line for line in BRAND)
        return [
            ("class:brand.logo", logo + "\n"),
            ("class:brand.name", "  EmergePolicy"),
            ("class:brand.meta", f"  /  {self.workspace.name}  ·  {self.state.model}  ·  {self.state.status}"),
        ]

    def _footer(self):
        tokens = self.state.usage.get("total_tokens", 0)
        return (
            f" / commands  Ctrl+C quit/stop  Esc close menu"
            f"  ·  {tokens} tokens  {self.state.duration_ms / 1000:.1f}s"
        )

    def _restore_session(self):
        self.sessions.invalidate(self.state.session_id)
        session = self.sessions.get_or_create(self.state.session_id)
        for message in session.messages:
            if message["role"] in {"user", "assistant"} and message.get("content"):
                self.state.entries.append({
                    "role": message["role"], "text": message["content"],
                })
        if session.metadata.get("model"):
            self.state.model = session.metadata["model"]

    def _bindings(self):
        keys = KeyBindings()

        @keys.add("enter")
        def submit(event):
            if self.palette_open:
                choices = self._choices()
                if choices:
                    command = choices[self.palette_index % len(choices)][0]
                    self.close_palette()
                    self.command(command)
                return
            message = self.editor.text.strip()
            if not message:
                return
            if message.startswith("/"):
                self.editor.text = ""
                self.command(message)
            elif self.busy:
                self.state.note("Run in progress. Draft kept; use /stop before sending.")
                self.refresh()
            else:
                self.editor.text = ""
                self.cancel = asyncio.Event()
                self.task = self.application.create_background_task(
                    self.run_turn(message),
                )

        @keys.add("/", filter=Condition(lambda: not self.palette_open))
        def slash(event):
            if event.current_buffer is self.editor.buffer and not self.editor.text:
                self.open_palette(COMMANDS, query="/")
            else:
                event.current_buffer.insert_text("/")

        @keys.add("escape", "enter")
        @keys.add("c-j")
        def newline(event):
            if not self.palette_open:
                self.editor.buffer.insert_text("\n")

        @keys.add("up", filter=Condition(lambda: self.palette_open))
        def up(event):
            self.palette_index -= 1
            self.application.invalidate()

        @keys.add("down", filter=Condition(lambda: self.palette_open))
        def down(event):
            self.palette_index += 1
            self.application.invalidate()

        @keys.add("escape")
        def escape(event):
            if self.palette_open:
                self.close_palette()

        @keys.add("c-c")
        def control_c(event):
            if self.palette_open:
                self.close_palette()
            elif not self.busy:
                self.application.exit()
            elif self.cancel and self.cancel.is_set():
                self.closing = True
                self.state.note("Exit requested; waiting for controller stop confirmation.")
                self.refresh()
            else:
                self.stop()
        return keys

    def _palette_changed(self, _buffer):
        self.palette_index = 0
        self.application.invalidate()

    def _choices(self):
        query = self.palette_input.text.strip().lower()
        return [
            entry for entry in self.palette_entries
            if query in " ".join(entry).lower()
        ]

    def _palette_text(self):
        choices = self._choices()
        if not choices:
            return " No matching command"
        selected = self.palette_index % len(choices)
        start = max(0, selected - 11)
        return [
            (
                "class:selected" if index == selected else "class:palette",
                f" {'›' if index == selected else ' '} {command:<16} {title}\n",
            )
            for index, (command, title) in enumerate(
                choices[start:start + 14], start,
            )
        ]

    def open_palette(self, entries, query=""):
        self.palette_entries = entries
        self.palette_index = 0
        self.palette_open = True
        self.palette_input.text = query
        self.palette_input.buffer.cursor_position = len(query)
        self.application.layout.focus(self.palette_input)
        self.application.invalidate()

    def close_palette(self):
        self.palette_open = False
        self.application.layout.focus(self.editor)
        self.application.invalidate()

    def stop(self):
        if self.cancel:
            self.cancel.set()
            self.state.status = "cancelling · awaiting controller"
            self.refresh()

    def quit(self):
        if self.busy:
            self.closing = True
            self.stop()
        else:
            self.application.exit()

    def command(self, text):
        command, _, arg = text.partition(" ")
        if command in {"/new", "/resume", "/model"} and self.busy:
            self.state.note("Finish or stop this run before changing session/model.")
        elif command == "/new":
            reset_workspace_context(self.workspace)
            self.state = ViewState("cli:" + uuid4().hex, self.state.model)
            self.logs.clear()
            self.show_logs = False
            self.last_output = None
            self.snapshot = None
            self.snapshot_time = 0.0
            self.refresh(force_snapshot=True)
        elif command == "/sessions":
            sessions = self.sessions.list_sessions()
            if sessions:
                self.open_palette([
                    ("/resume " + item["key"], item["title"]) for item in sessions
                ], query="/resume ")
            else:
                self.state.note("No saved sessions in this workspace.")
        elif command == "/resume":
            known = {item["key"] for item in self.sessions.list_sessions()}
            if arg not in known:
                self.state.note("Session not found.")
            else:
                self.state = ViewState(arg, self.state.model)
                self._restore_session()
        elif command == "/model":
            if arg.strip():
                self.state.model = arg.strip()
            else:
                self.editor.text = "/model "
                self.editor.buffer.cursor_position = len(self.editor.text)
        elif command == "/stop":
            self.stop()
        elif command == "/quit":
            self.quit()
        elif command in {"/commands", "/help"}:
            self.open_palette(COMMANDS, query="/")
        elif command == "/details":
            self.details = not self.details
        elif command == "/logs":
            self.show_logs = not self.show_logs
        elif command == "/theme":
            self.light = not self.light
            self.application.style = style(self.light)
            self._save_preferences()
        elif command == "/sidebar":
            self.sidebar = not self.sidebar
            self._save_preferences()
        elif command == "/refresh":
            self.refresh(force_snapshot=True)
        elif command == "/health":
            self.health = [{"name": "Services", "status": "checking"}]
            self.application.create_background_task(self.probe_health())
        elif command == "/export":
            path = self.workspace / "exports" / (uuid4().hex + ".md")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.render_transcript(), encoding="utf-8")
            self.state.note(f"Exported: {path}")
        elif command == "/workspace":
            self.state.note(f"Workspace: {self.workspace}")
        elif command == "/observations":
            self.state.note(
                "Observation manifest: "
                + str(self.workspace / "artifacts/observations/observation.json")
            )
        elif command == "/runs":
            self.state.note(
                f"Latest run: {self.last_output}" if self.last_output
                else "No run has started in this UI."
            )
        elif command == "/keys":
            self.state.note(
                "Enter sends · Alt+Enter/Ctrl+J inserts a newline · / opens commands "
                "· Ctrl+C exits when idle or requests stop while running."
            )
        else:
            self.state.note("Unknown command. Press / at an empty prompt.")
        self.dirty = True
        self.refresh()

    def _save_preferences(self):
        atomic_json(
            self.preferences_path,
            {"light": self.light, "sidebar": self.sidebar},
        )

    async def probe_health(self):
        self.health = await service_health(self.config)
        self.dirty = True

    async def run_turn(self, message):
        self.cancel = self.cancel or asyncio.Event()
        request = RunRequest(
            message=message,
            session_id=self.state.session_id,
            config=self.config_path,
            workspace=str(self.workspace),
            model=self.state.model,
            stream=True,
        )
        self.last_output = self.workspace / "runs" / request.run_id
        try:
            result = await self.runtime.run(
                request, output_dir=self.last_output,
                on_event=self.on_event, cancel=self.cancel,
            )
            if self.closing and result.finish_reason != "cancellation_unconfirmed":
                self.application.exit()
            elif self.closing:
                self.closing = False
                self.state.note(
                    "Stop was NOT confirmed. Check the controller before exiting.",
                )
        except Exception as exc:
            self.state.status = "failed"
            self.state.note(f"Runtime/artifact error: {exc}")
            self.closing = False
        finally:
            self.cancel = None
            self.dirty = True
            self.refresh(force_snapshot=True)

    def on_event(self, event):
        self.state.accept(event)
        if event.type in {"plan.updated", "action.updated", "run.finished"}:
            self.snapshot_time = 0.0
        self.dirty = True

    def _render_entry(self, entry, width):
        role = entry["role"]
        signature = (
            role, entry.get("text"), entry.get("streaming"), self.details, width,
            entry.get("tool"), entry.get("status"), entry.get("duration_ms"),
            repr(entry.get("arguments")) if self.details else None,
            entry.get("result") if self.details else None,
        )
        key = id(entry)
        cached = self._entry_cache.get(key)
        if cached and cached[0] == signature:
            return cached[1]

        if role == "tool":
            parts = [
                f"  ↳ {entry['tool']} · {entry['status']} "
                f"· {entry.get('duration_ms', 0)}ms"
            ]
            if self.details:
                parts.append(json.dumps(
                    entry.get("arguments", {}), ensure_ascii=False, indent=2,
                ))
                parts.append(entry.get("result", ""))
            rendered = "\n".join(parts)
        elif role == "system":
            rendered = "· " + entry["text"]
        elif not entry["text"]:
            rendered = ""
        else:
            label = "YOU" if role == "user" else "EMERGE"
            if entry.get("streaming"):
                body = entry["text"]
            else:
                stream = io.StringIO()
                Console(
                    file=stream, width=width, color_system=None,
                ).print(Markdown(entry["text"]))
                body = stream.getvalue().rstrip()
            rendered = label + "\n" + body

        self._entry_cache[key] = (signature, rendered)
        return rendered

    def render_transcript(self):
        width = max(
            30,
            self.application.output.get_size().columns - (
                self.SIDEBAR_WIDTH + 6 if self.sidebar else 8
            ),
        )
        live_entries = {id(entry) for entry in self.state.entries}
        self._entry_cache = {
            key: value for key, value in self._entry_cache.items()
            if key in live_entries
        }
        blocks = [
            rendered for entry in self.state.entries
            if (rendered := self._render_entry(entry, width))
        ]
        return "\n\n".join(blocks) or (
            "E M E R G E\n\nYour robot workspace.\n\n"
            "Describe a task to begin, or press / to explore commands."
        )

    def _status_style(self, status):
        value = str(status or "unknown").lower()
        return "status." + (
            value if value in {
                "ready", "pending", "running", "completed", "failed",
                "cancelled", "unknown",
            } else "unknown"
        )

    @staticmethod
    def _bool_style(value):
        return "sidebar.good" if value is True else "sidebar.bad" if value is False else "sidebar.value"

    def _build_sidebar(self, snapshot):
        fragments = []

        def row(*parts):
            for token, text in parts:
                fragments.append(("class:" + token, str(text)))
            fragments.append(("", "\n"))

        def section(title, token):
            if fragments:
                row(("sidebar", ""))
            row((token, "◆ " + title))

        section("SESSION", "sidebar.section.session")
        row(("sidebar.value", self.state.session_id))

        section("PLAN", "sidebar.section.plan")
        plan = snapshot["plan"]
        if plan.get("error"):
            row(("error", "Unreadable: " + plan["error"]))
        else:
            row(("sidebar.label", "Mission  "), ("sidebar.value", plan.get("mission") or "No plan"))
            items = plan.get("main_line", [])
            done = sum(item.get("status") == "done" for item in items)
            if items:
                filled = round(12 * done / len(items))
                bar = "━" * filled + "─" * (12 - filled)
                row(
                    ("sidebar.label", "Progress "),
                    ("plan.progress", f"{done}/{len(items)} "),
                    ("plan.progress", bar),
                )
            for item in items:
                current = item.get("id") == plan.get("pointer")
                token = (
                    "plan.current" if current
                    else "plan.done" if item.get("status") == "done"
                    else "plan.pending"
                )
                marker = "▶" if current else "✓" if item.get("status") == "done" else "○"
                row((token, f" {marker} {item.get('id', '?')}. {item.get('subgoal', '?')} "))

        section("ROBOT", "sidebar.section.robot")
        robot = snapshot["robot"]
        if robot["error"]:
            row(("error", "Unreadable: " + robot["error"]))
        elif robot["age_s"] is None:
            row(("sidebar.warn", "No snapshot"))
        else:
            row(("sidebar.label", "Age  "), ("sidebar.count", f"{robot['age_s']:.0f}s"))
            robots = robot["data"].get("robots", {})
            for name, data in robots.items():
                row(("robot.name", str(name)))
                if isinstance(data, dict):
                    for key in ("connected", "success", "done"):
                        if key in data:
                            row(
                                ("sidebar.label", f"  {key:<10}"),
                                (self._bool_style(data[key]), str(data[key])),
                            )

        actions = snapshot["actions"]
        if actions["error"]:
            row(("error", "Actions unreadable: " + actions["error"]))
        else:
            recent_actions = actions["data"].get("actions", [])[-3:]
            if not recent_actions:
                row(("sidebar.label", "No actions"))
            for action in recent_actions:
                status = action.get("status", "unknown")
                row(
                    ("action.name", str(action.get("action_type", "?"))),
                    ("sidebar.label", "  ·  "),
                    (self._status_style(status), str(status)),
                )

        section("OBSERVATION", "sidebar.section.observation")
        observation = snapshot["observation"]
        summary = observation["summary"]
        if observation["error"]:
            row(("error", "Unreadable: " + observation["error"]))
        elif summary["status"] == "missing":
            row(("sidebar.warn", "No observation manifest"))
        else:
            status = summary["status"]
            status_token = "sidebar.good" if status == "ready" else "sidebar.bad"
            row(("sidebar.label", "Status     "), (status_token, status))
            row(
                ("sidebar.label", "Revision   "),
                ("sidebar.count", str(summary.get("revision", "?"))),
            )
            row(
                ("sidebar.label", "Reference  "),
                ("observation.reference", summary.get("reference_view", "?")),
            )
            row(
                ("sidebar.label", "Views      "),
                ("sidebar.count", str(summary.get("view_count", 0))),
            )
            row(
                ("sidebar.label", "Age        "),
                ("sidebar.value", f"{observation['age_s']:.0f}s"),
            )
            if summary.get("missing_images"):
                row(
                    ("error", f"Missing images: {len(summary['missing_images'])}"),
                )

        section("SERVICES", "sidebar.section.services")
        if not self.health:
            row(("sidebar.warn", "Not probed  ·  /health"))
        for item in self.health:
            name = str(item["name"])
            name_token = (
                "service.vggt" if name.upper() == "VGGT"
                else "service.sam3" if name.upper() == "SAM3"
                else "sidebar.value"
            )
            row(
                (name_token, name),
                ("sidebar.label", "  ·  "),
                (self._status_style(item["status"]), item["status"]),
            )

        if self.last_output:
            section("RUN ARTIFACTS", "sidebar.section.artifacts")
            row(("sidebar.path", self.last_output))
        return fragments

    def refresh(self, *, force_snapshot=False):
        content = "\n".join(self.logs) if self.show_logs else self.render_transcript()
        at_end = self.transcript.buffer.cursor_position == len(self.transcript.text)
        cursor = (
            len(content) if at_end
            else min(self.transcript.buffer.cursor_position, len(content))
        )
        self.transcript.buffer.set_document(
            Document(content, cursor), bypass_readonly=True,
        )
        now = time.monotonic()
        if (
            force_snapshot
            or self.snapshot is None
            or now - self.snapshot_time >= self.SNAPSHOT_INTERVAL_S
        ):
            self.snapshot = workspace_snapshot(self.workspace)
            self.snapshot_time = now
        self.sidebar_fragments = self._build_sidebar(self.snapshot)
        self.dirty = False
        self.application.invalidate()

    async def run(self):
        logger.remove()

        def capture_log(message):
            self.logs.append(str(message).rstrip())
            self.dirty = True

        sink = logger.add(capture_log, backtrace=False, diagnose=False)

        async def tick():
            while True:
                await asyncio.sleep(0.03)
                if (
                    self.dirty
                    or time.monotonic() - self.snapshot_time
                    >= self.SNAPSHOT_INTERVAL_S
                ):
                    self.refresh()

        ticker = asyncio.create_task(tick())
        try:
            with patch_stdout(raw=True):
                await self.application.run_async()
        finally:
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
            logger.remove(sink)


def launch(**kwargs):
    asyncio.run(WorkspaceApp(**kwargs).run())
