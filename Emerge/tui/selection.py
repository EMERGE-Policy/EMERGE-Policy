"""Mouse selection and clipboard writes for the full-screen client."""

import asyncio
import base64
import os
import shutil
import subprocess
import sys
import time

from prompt_toolkit.application import get_app
from prompt_toolkit.clipboard import InMemoryClipboard
from prompt_toolkit.document import Document
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.mouse_events import MouseButton, MouseEventType
from prompt_toolkit.widgets import TextArea


class TerminalClipboard(InMemoryClipboard):
    """Copy through the terminal (including SSH) and the local desktop when available."""

    def __init__(self):
        super().__init__()
        self.notice_until = 0.0
        self.notice = ""
        self._lock = asyncio.Lock()
        self.command = None
        if not (os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")):
            if sys.platform == "darwin" and shutil.which("pbcopy"):
                self.command = ["pbcopy"]
            elif os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
                self.command = ["wl-copy"]
            elif os.environ.get("DISPLAY") and shutil.which("xclip"):
                self.command = ["xclip", "-selection", "clipboard"]
            elif os.environ.get("DISPLAY") and shutil.which("xsel"):
                self.command = ["xsel", "--clipboard", "--input"]

    def set_data(self, data):
        super().set_data(data)
        app = get_app()
        encoded = base64.b64encode(data.text.encode("utf-8")).decode("ascii")
        sequence = f"\x1b]52;c;{encoded}\x07"
        if os.environ.get("TMUX"):
            sequence += "\x1bPtmux;\x1b" + sequence + "\x1b\\"
        app.output.write_raw(sequence)
        app.output.flush()
        self.notice = "Copy sent to terminal"
        self.notice_until = time.monotonic() + 3
        if self.command:
            app.create_background_task(self._write_local(data.text))
        app.invalidate()

    async def _write_local(self, text):
        async with self._lock:
            try:
                await asyncio.to_thread(
                    subprocess.run, self.command, input=text.encode("utf-8"),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=2, check=True,
                )
            except (OSError, subprocess.SubprocessError):
                return  # The terminal copy was already sent.
            self.notice = "Copied selection"
            get_app().invalidate()


class FragmentLexer(Lexer):
    def __init__(self, lines):
        self.lines = lines

    def lex_document(self, document):
        return self.lines.__getitem__


class CopyTextArea(TextArea):
    """Select on the first drag and copy on release without leaving the TUI."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.focus_after_copy = None
        self._default_mouse_handler = self.control.mouse_handler
        self.control.mouse_handler = self._mouse_handler

    def _mouse_handler(self, event):
        app = get_app()
        if event.event_type in {MouseEventType.SCROLL_UP, MouseEventType.SCROLL_DOWN}:
            self.buffer.exit_selection()
        if event.button == MouseButton.LEFT and event.event_type == MouseEventType.MOUSE_DOWN:
            for control in app.layout.find_all_controls():
                if isinstance(control, BufferControl):
                    control.buffer.exit_selection()
            app.layout.focus(self.control)
        result = self._default_mouse_handler(event)
        if event.button == MouseButton.LEFT and event.event_type == MouseEventType.MOUSE_UP:
            self.copy_selection()
            if self.focus_after_copy is not None:
                app.layout.focus(self.focus_after_copy)
        return result

    def copy_selection(self):
        if self.buffer.selection_state is None:
            return False
        _, data = self.buffer.document.cut_selection()
        if not data.text:
            return False
        get_app().clipboard.set_data(data)
        return True

    def set_formatted_text(self, fragments):
        """Keep styled headers and sidebar text selectable, without losing their colors."""
        if self.buffer.selection_state is not None:
            return
        lines = [[]]
        for style, text in fragments:
            for index, part in enumerate(text.split("\n")):
                if index:
                    lines.append([])
                lines[-1].append((style, part))
        self.lexer = FragmentLexer(lines)
        text = "".join(text for _, text in fragments)
        self.buffer.set_document(
            Document(text, min(self.buffer.cursor_position, len(text))), bypass_readonly=True,
        )
