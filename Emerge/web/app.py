"""Local Web API and static frontend for Emerge's existing runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from Emerge.config.setup import SetupStatus, SetupValues, complete_setup, describe_setup
from Emerge.providers.registry import find_by_name
from Emerge.runtime.controller_control import read_json
from Emerge.runtime.snapshots import service_health
from Emerge.web.service import WebService


class CreateRequest(BaseModel):
    scene_id: str = Field(alias="sceneId")


class SendRequest(BaseModel):
    message: str = Field(min_length=1)
    model: str | None = None


class OperationRequest(BaseModel):
    operation: Literal["stop", "reset", "switch", "close", "restart", "delete", "archive"]
    scene_id: str | None = Field(default=None, alias="sceneId")


class UpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    model: str | None = Field(default=None, min_length=1)
    archived: Literal[False] | None = None


def create_app(service: WebService, frontend: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        app.state.snapshot = service.snapshot()

        async def observe():
            while True:
                app.state.snapshot = service.snapshot()
                await asyncio.sleep(0.5)

        monitor = asyncio.create_task(observe())
        try:
            yield
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            await service.shutdown()

    app = FastAPI(title="EMERGE Web", lifespan=lifespan)

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).hostname != request.url.hostname:
            return JSONResponse({"detail": "Origin does not match this server"}, status_code=403)
        if (
            service.setup_required
            and request.method in {"POST", "PATCH", "DELETE"}
            and request.url.path.startswith("/api/instances")
        ):
            return JSONResponse({"detail": "Complete workspace setup first"}, status_code=409)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def conflict(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    def instance(key: str):
        if key not in service.instances:
            raise HTTPException(404, "Instance not found")
        return service.instances[key]

    @app.get("/api/setup", response_model=SetupStatus)
    async def setup():
        return JSONResponse(
            describe_setup(service.config, service.config_path).model_dump(by_alias=True),
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/setup", response_model=SetupStatus)
    async def configure(body: SetupValues):
        if not service.setup_required:
            raise ValueError("Workspace is already configured. Reload to continue.")
        spec = find_by_name(body.provider)
        if spec and spec.is_oauth:
            raise ValueError("Use the terminal setup for OAuth providers, then restart the Web service.")
        try:
            service.config = complete_setup(service.config, service.config_path, body)
        except OSError:
            raise HTTPException(500, "Could not save configuration. Check file permissions and retry.") from None
        status = describe_setup(service.config, service.config_path)
        service.setup_required = status.required
        return JSONResponse(status.model_dump(by_alias=True), headers={"Cache-Control": "no-store"})

    @app.get("/api/catalog")
    async def catalog():
        return {
            "root": service.catalog["root"],
            "scenes": [
                {
                    "id": entry["id"],
                    "name": Path(entry["label"]).stem.replace("_", " "),
                    "group": str(Path(entry["label"]).parent),
                    "description": entry["label"],
                }
                for entry in service.catalog["entries"]
            ],
        }

    @app.get("/api/state")
    async def state():
        return service.snapshot()

    @app.get("/api/events")
    async def events(request: Request):
        # Authoritative projections use stable message IDs. Reconnect replaces state;
        # it never appends a second copy of historical messages or replays instructions.
        async def stream():
            while not await request.is_disconnected():
                yield "data: " + json.dumps(app.state.snapshot, ensure_ascii=False) + "\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/instances", status_code=202)
    async def create(body: CreateRequest):
        return service.create(body.scene_id)

    @app.patch("/api/instances/{key}")
    async def update(key: str, body: UpdateRequest):
        item = instance(key)
        service.require_idle_operation(key)
        item.update(body.model_dump(exclude_none=True))
        service.save(item)
        return item

    @app.post("/api/instances/{key}/runs", status_code=202)
    async def send(key: str, body: SendRequest):
        item = instance(key)
        async with service.locks.setdefault(key, asyncio.Lock()):
            run_id = await service.send(item, body.message, body.model)
        return {"runId": run_id}

    @app.post("/api/instances/{key}/operations", status_code=202)
    async def operate(key: str, body: OperationRequest):
        item = instance(key)
        service.require_idle_operation(key)
        if body.operation == "switch":
            if body.scene_id not in {entry["id"] for entry in service.catalog["entries"]}:
                raise ValueError("Select a scene from the driver catalog")
            if item["status"] == "ready" and item["sceneId"] == body.scene_id:
                return item
        item["operation"] = body.operation
        service.save(item)
        service.schedule(item, service.operate(item, body.operation, body.scene_id))
        return item

    def camera_image(key: str, camera: str) -> Path:
        instance(key)
        workspace = service.workspace(key)
        manifest = read_json(workspace / "artifacts/observations/observation.json")
        view = next(
            (view for view in (manifest or {}).get("views", []) if view["name"] == camera), None
        )
        if view is None:
            raise HTTPException(404, "Camera observation is not available")
        path = (workspace / view["image_path"]).resolve()
        if not path.is_relative_to((workspace / "artifacts").resolve()) or not path.is_file():
            raise HTTPException(404, "Camera image is not available in this workspace")
        return path

    @app.get("/api/instances/{key}/image")
    async def image(key: str, camera: str):
        path = camera_image(key, camera)
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.websocket("/api/instances/{key}/image-stream")
    async def image_stream(websocket: WebSocket, key: str, camera: str):
        origin = websocket.headers.get("origin")
        if origin and urlparse(origin).hostname != websocket.url.hostname:
            await websocket.close(code=1008)
            return
        try:
            camera_image(key, camera)
        except HTTPException:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        previous = None
        loop = asyncio.get_running_loop()
        try:
            while key in service.instances and not service.closing:
                # The browser requests another frame only after displaying the last.
                # No per-client frame queue: always read the newest atomic PNG.
                await websocket.receive_text()
                deadline = loop.time() + 1
                while loop.time() < deadline:
                    try:
                        path = camera_image(key, camera)
                        revision = (path, path.stat().st_mtime_ns)
                        if revision != previous:
                            data = await asyncio.to_thread(path.read_bytes)
                            await websocket.send_bytes(data)
                            previous = revision
                            break
                    except (FileNotFoundError, HTTPException):
                        # A scene reset briefly removes observation files.
                        pass
                    await asyncio.sleep(0.01)
                else:
                    # Keep idle connections responsive to disconnects and retries.
                    await websocket.send_text("")
            await websocket.close()
        except WebSocketDisconnect:
            pass

    @app.get("/api/instances/{key}/files")
    async def files(key: str):
        instance(key)
        directory = service.workspace(key).parent
        return [
            {"path": path.relative_to(directory).as_posix(), "size": path.stat().st_size}
            for root in (
                directory / "logs",
                directory / "workspace/runs",
                directory / "workspace/artifacts",
            )
            for path in root.rglob("*")
            if path.is_file() and path.resolve().is_relative_to(directory)
        ]

    @app.get("/api/instances/{key}/file")
    async def file(key: str, path: str):
        instance(key)
        directory = service.workspace(key).parent
        target = (directory / path).resolve()
        allowed = (
            directory / "logs",
            directory / "workspace/runs",
            directory / "workspace/artifacts",
        )
        if not any(target.is_relative_to(root) for root in allowed) or not target.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(target, filename=target.name)

    @app.get("/api/instances/{key}/logs")
    async def logs(key: str, kind: Literal["agent", "controller"] = "agent"):
        instance(key)
        path = service.workspace(key).parent / "logs" / f"{kind}.log"
        if not path.exists():
            return {"text": "No log output yet."}
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 100_000))
            return {"text": stream.read().decode("utf-8", errors="replace")}

    @app.get("/api/settings")
    async def settings():
        config = service.config
        models = {
            config.agents.defaults.model,
            *(item["model"] for item in service.instances.values()),
        }
        return {
            "models": sorted(models),
            "defaultModel": config.agents.defaults.model,
            "driver": service.driver,
            "dataDirectory": str(service.root),
            "python": sys.executable,
        }

    @app.get("/api/health")
    async def health():
        return await service_health(service.config)

    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


def main(argv=None):
    import uvicorn

    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="EMERGE browser workspace")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".Emerge/web")
    parser.add_argument("--driver", default="libero_mujoco")
    parser.add_argument(
        "--driver-config", type=Path, default=repo / "dev/libero_mujoco_driver_sample.json"
    )
    parser.add_argument("--frontend", type=Path, default=repo / "web/dist")
    args = parser.parse_args(argv)
    service = WebService(
        args.data_dir.expanduser(),
        repo,
        args.config.expanduser().resolve() if args.config else None,
        args.driver,
        args.driver_config.expanduser(),
    )
    uvicorn.run(
        create_app(service, args.frontend),
        host=args.host,
        port=args.port,
        ws_per_message_deflate=False,
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
