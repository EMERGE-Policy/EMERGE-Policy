#!/usr/bin/env python3
"""Live MJPEG grid stream of concurrent LIBERO episodes for browser viewing."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

TILE_SIZE = 256  # agentview renders at 256x256; tiles are normalized to this.
_LABEL_BAR = 22
_STATUS_COLORS = {
    "running": (80, 200, 80),
    "success": (60, 200, 60),
    "failed": (60, 60, 220),
    "idle": (90, 90, 90),
}


def grid_dims(n: int) -> tuple[int, int]:
    """Return (rows, cols) for an n-tile near-square grid: 8->3x3, 4->2x2."""
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    return rows, cols


@dataclass
class _Slot:
    manifest_path: Path | None = None
    view_name: str = ""
    label: str = ""
    status: str = "idle"
    last_frame: np.ndarray | None = None


class StreamBoard:
    """Thread-safe registry of worker slots and their latest-frame snapshots."""

    def __init__(self, slots: int) -> None:
        self._lock = threading.Lock()
        self._slots: dict[int, _Slot] = {i: _Slot() for i in range(max(1, slots))}
        self._rows, self._cols = grid_dims(len(self._slots))

    def register(
        self,
        slot: int,
        manifest_path: Path,
        view_name: str,
        label: str,
    ) -> None:
        with self._lock:
            self._slots[slot] = _Slot(
                manifest_path=Path(manifest_path),
                view_name=view_name,
                label=label,
                status="running",
            )

    def update_status(self, slot: int, status: str) -> None:
        with self._lock:
            if slot in self._slots:
                self._slots[slot].status = status

    def release(self, slot: int) -> None:
        with self._lock:
            if slot in self._slots:
                self._slots[slot] = _Slot()

    def _read_tile(self, slot: _Slot) -> np.ndarray:
        """Read a slot's latest published view, reusing the last visible frame."""
        frame: np.ndarray | None = None
        image_path = self._latest_view_path(slot)
        if image_path is not None and image_path.exists():
            img = cv2.imread(str(image_path))
            if img is not None:
                frame = cv2.resize(img, (TILE_SIZE, TILE_SIZE))
                slot.last_frame = frame
        if frame is None:
            frame = (
                slot.last_frame
                if slot.last_frame is not None
                else np.zeros((TILE_SIZE, TILE_SIZE, 3), np.uint8)
            )
        return self._decorate(frame.copy(), slot)

    @staticmethod
    def _latest_view_path(slot: _Slot) -> Path | None:
        manifest_path = slot.manifest_path
        if manifest_path is None or not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        view = next(
            item for item in manifest["views"] if item["name"] == slot.view_name
        )
        workspace = manifest_path.parents[2]
        return workspace / view["image_path"]

    @staticmethod
    def _decorate(tile: np.ndarray, slot: _Slot) -> np.ndarray:
        color = _STATUS_COLORS.get(slot.status, _STATUS_COLORS["idle"])
        cv2.rectangle(tile, (0, 0), (TILE_SIZE - 1, TILE_SIZE - 1), color, 2)
        cv2.rectangle(tile, (0, 0), (TILE_SIZE, _LABEL_BAR), (0, 0, 0), -1)
        text = slot.label or "(idle)"
        cv2.putText(
            tile, text[:34], (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA,
        )
        return tile

    def render_grid(self) -> np.ndarray:
        with self._lock:
            tiles = [self._read_tile(self._slots[i]) for i in sorted(self._slots)]
        blank = np.zeros((TILE_SIZE, TILE_SIZE, 3), np.uint8)
        while len(tiles) < self._rows * self._cols:
            tiles.append(blank)
        rows = [
            np.hstack(tiles[r * self._cols:(r + 1) * self._cols])
            for r in range(self._rows)
        ]
        return np.vstack(rows)

    def render_jpeg(self, quality: int = 80) -> bytes | None:
        ok, buf = cv2.imencode(
            ".jpg", self.render_grid(), [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return buf.tobytes() if ok else None


_INDEX_HTML = b"""<!doctype html><html><head><meta charset=utf-8>
<title>LIBERO eval stream</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;
align-items:center;height:100vh}img{max-width:100vw;max-height:100vh;
image-rendering:pixelated}</style></head>
<body><img src="/stream.mjpg"></body></html>"""

_BOUNDARY = "libero-frame"


class _Handler(BaseHTTPRequestHandler):
    board: StreamBoard
    fps: float

    def log_message(self, *_args: Any) -> None:  # silence per-request logging
        pass

    def do_GET(self) -> None:  # noqa: N802 (required by BaseHTTPRequestHandler)
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_INDEX_HTML)))
            self.end_headers()
            self.wfile.write(_INDEX_HTML)
        elif self.path.startswith("/stream.mjpg"):
            self._serve_stream()
        else:
            self.send_error(404)

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
        )
        self.send_header("Cache-Control", "no-cache, private")
        self.end_headers()
        period = 1.0 / max(1.0, self.fps)
        try:
            while True:
                jpeg = self.board.render_jpeg()
                if jpeg is not None:
                    self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                threading.Event().wait(period)
        except (BrokenPipeError, ConnectionResetError):
            return  # client closed the tab; not an error


class StreamServer:
    """Run the MJPEG grid server on a daemon thread."""

    def __init__(
        self, board: StreamBoard, *, host: str = "127.0.0.1", port: int = 8008,
        fps: float = 10.0,
    ) -> None:
        handler = type(
            "_BoundHandler", (_Handler,), {"board": board, "fps": float(fps)}
        )
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="libero-stream", daemon=True
        )

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
