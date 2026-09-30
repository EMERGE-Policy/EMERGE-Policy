"""Own isolated Controllers and headless AgentRuntime runs for browser conversations."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path
from uuid import uuid4

import psutil

from Emerge.config.setup import describe_setup, load_setup
from Emerge.runtime.controller_control import (
    control_lock,
    expired,
    new_reset_request,
    new_scene_request,
    read_json,
    result_file,
    update_request,
    write_request,
    write_result,
)
from Emerge.runtime.protocol import RunEvent, RunRequest
from Emerge.runtime.snapshots import workspace_snapshot
from Emerge.runtime.storage import WorkspaceLease, atomic_json
from Emerge.runtime.workspace import reset_workspace_context
from Emerge.utils.action_queue import cancel_actions
from Emerge.utils.helpers import sync_workspace_templates
from Emerge.web.events import RunView
from robot.controller import load_driver_config


class WebService:
    def __init__(
        self, root: Path, repo: Path, config: Path | None, driver: str, driver_config: Path
    ):
        self.root, self.repo = root.resolve(), repo.resolve()
        self.config_path, self.config = load_setup(config)
        self.setup_required = describe_setup(self.config, self.config_path).required
        self.driver, self.driver_config = driver, driver_config.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "state.sqlite3")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS instances (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self.instances = {
            key: json.loads(data) for key, data in self.db.execute("SELECT id, data FROM instances")
        }
        self.controllers: dict[str, asyncio.subprocess.Process] = {}
        self.agents: dict[str, asyncio.subprocess.Process] = {}
        self.operations: dict[str, asyncio.Task] = {}
        self.views: dict[str, RunView] = {}
        self.offsets: dict[str, int] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.capacity_lock = asyncio.Lock()
        self.notice = ""
        self.catalog = {"root": "", "entries": []}
        self.closing = False

    def workspace(self, key: str) -> Path:
        return self.root / "instances" / key / "workspace"

    def save(self, item: dict):
        self.db.execute(
            "INSERT OR REPLACE INTO instances VALUES (?, ?)", (item["id"], json.dumps(item))
        )
        self.db.commit()

    async def start(self):
        self.lease = WorkspaceLease(self.root)
        self.lease.__enter__()
        # PID plus creation time identifies surviving children without touching unrelated processes.
        for item in self.instances.values():
            item["operation"] = None
            for field in ("agent", "controller"):
                identity = item.get(field)
                if identity:
                    await self.terminate_identity(identity)
                    item[field] = None
            if item["status"] != "closed":
                item.update(
                    status="error",
                    error="Web service restarted. Previous processes were reclaimed; explicitly restart the scene.",
                    runStatus="failed",
                )
            self.save(item)
        catalog_path = self.root / "catalog.json"
        with (self.root / "catalog.log").open("ab") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "Emerge.web.catalog",
                self.driver,
                str(self.driver_config),
                str(catalog_path),
                cwd=self.repo,
                stdout=log,
                stderr=log,
            )
            code = await process.wait()
        if code:
            raise RuntimeError(f"Cannot load scene catalog. See {self.root / 'catalog.log'}")
        self.catalog = read_json(catalog_path)
        for item in self.instances.values():
            self.views[item["id"]] = RunView()
            for run_id in item.get("runs", []):
                self.consume_events(item, run_id)

    async def terminate_identity(self, identity: dict):
        try:
            process = psutil.Process(identity["pid"])
            if abs(process.create_time() - identity["created"]) > 0.01:
                return
            children = process.children(recursive=True)
            if process.status() == psutil.STATUS_ZOMBIE:
                return
            os.killpg(process.pid, signal.SIGINT)
            _, alive = await asyncio.to_thread(psutil.wait_procs, [process, *children], timeout=15)
            for child in alive:
                child.kill()
            _, alive = await asyncio.to_thread(psutil.wait_procs, alive, timeout=5)
            if alive:
                raise RuntimeError("Managed processes have not stopped")
        except psutil.NoSuchProcess:
            return

    async def spawn(self, item: dict, kind: str, command: list[str]):
        directory = self.workspace(item["id"]).parent
        (directory / "logs").mkdir(exist_ok=True)
        env = dict(os.environ)
        # Per-run configuration lives in each child; API workers never mutate process-wide model state.
        with (directory / "logs" / f"{kind}.log").open("ab") as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.repo,
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        item[kind] = {"pid": process.pid, "created": psutil.Process(process.pid).create_time()}
        self.save(item)
        return process

    async def reclaim(self, item: dict, kind: str):
        registry = self.controllers if kind == "controller" else self.agents
        process = registry.get(item["id"])
        if process:
            if process.returncode is None:
                descendants = psutil.Process(process.pid).children(recursive=True)
                os.killpg(process.pid, signal.SIGINT)
                try:
                    await asyncio.wait_for(asyncio.shield(process.wait()), 15)
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await asyncio.wait_for(asyncio.shield(process.wait()), 5)
                for child in descendants:
                    if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                        child.kill()
                _, alive = await asyncio.to_thread(psutil.wait_procs, descendants, timeout=5)
                if alive:
                    raise RuntimeError("Managed child processes have not stopped")
        elif item.get(kind):
            await self.terminate_identity(item[kind])
        registry.pop(item["id"], None)
        item[kind] = None
        self.save(item)

    def schedule(self, item: dict, coroutine):
        async def work():
            try:
                await coroutine
            except Exception as exc:
                item.update(status="error", error=str(exc))
                if item["id"] in self.instances:
                    self.save(item)
            finally:
                if item["id"] in self.instances:
                    item["operation"] = None
                    self.save(item)

        task = asyncio.create_task(work())
        self.operations[item["id"]] = task
        return task

    def require_idle_operation(self, key: str):
        if self.closing or self.instances[key].get("operation"):
            raise ValueError("An environment operation is still in progress")
        task = self.operations.get(key)
        if task and not task.done():
            raise ValueError("An environment operation is still in progress")

    async def reserve(self, item: dict):
        active = [
            other
            for other in self.instances.values()
            if other["id"] != item["id"] and other.get("controller")
        ]
        if len(active) >= 10:
            oldest = min(active, key=lambda entry: entry["openedAt"])
            self.notice = (
                f"At capacity: deleting {oldest['title']} before starting {item['title']}."
            )
            operation = self.operations.get(oldest["id"])
            if operation and not operation.done():
                raise ValueError(
                    "The oldest instance is changing environment. Wait for it before creating another instance."
                )
            oldest["operation"] = "delete"
            async with self.locks.setdefault(oldest["id"], asyncio.Lock()):
                await self.close(oldest, delete=True)

    def create(self, scene_id: str):
        entry = next((entry for entry in self.catalog["entries"] if entry["id"] == scene_id), None)
        if entry is None:
            raise ValueError("Select a scene from the driver catalog")
        key, now = uuid4().hex, time.time() * 1000
        item = dict(
            id=key,
            title=Path(entry["label"]).stem.replace("_", " "),
            sceneId=scene_id,
            initialSceneId=scene_id,
            sessionId="web:" + uuid4().hex,
            openedAt=now,
            createdAt=now,
            status="queued",
            operation="start",
            runStatus="idle",
            model=self.config.agents.defaults.model,
            archived=False,
            runs=[],
            agent=None,
            controller=None,
            error=None,
        )
        self.instances[key] = item
        self.views[key] = RunView()
        self.save(item)
        self.schedule(item, self.launch(item))
        return item

    async def launch(self, item: dict):
        async with self.capacity_lock:
            await self.reserve(item)
            workspace = self.workspace(item["id"])
            workspace.mkdir(parents=True, exist_ok=True)
            if item.get("controller"):
                await self.stop(item)
                await self.reclaim(item, "controller")
            reset_workspace_context(workspace)
            sync_workspace_templates(workspace, silent=True)
            config = copy.deepcopy(load_driver_config(self.driver_config))
            config["workspace"] = str(workspace)
            if self.driver == "libero_mujoco":
                libero = config["libero"]
                root = Path(libero.get("bddl_root") or Path(libero["bddl_file_name"]).parent)
                libero["bddl_root"] = str((self.repo / root).resolve())
                libero["bddl_file_name"] = item["sceneId"]
                for name, camera in config.get("cameras", {}).items():
                    if "observation" in camera:
                        camera["observation"]["directory"] = f"artifacts/cameras/{name}"
                    if "recording" in camera:
                        camera["recording"]["path"] = f"artifacts/cameras/{name}.mp4"
            driver_path = workspace.parent / "controller.json"
            atomic_json(driver_path, config)
            item.update(
                status="starting",
                error=None,
                openedAt=time.time() * 1000,
                runStatus="idle",
                initialSceneId=item["sceneId"],
                sessionId="web:" + uuid4().hex,
                runs=[],
            )
            self.views[item["id"]] = RunView()
            self.controllers[item["id"]] = await self.spawn(
                item,
                "controller",
                [
                    sys.executable,
                    "-m",
                    "robot.controller",
                    "--driver",
                    self.driver,
                    "--workspace",
                    str(workspace),
                    "--driver-config",
                    str(driver_path),
                    "--interval",
                    "0.5",
                ],
            )
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                process = self.controllers[item["id"]]
                if process.returncode is not None:
                    raise RuntimeError(
                        f"Controller exited ({process.returncode}); see Controller log"
                    )
                status = read_json(workspace / ".controller/scene.json")
                if status and status.get("pid") == process.pid and status.get("ready"):
                    observation = workspace_snapshot(workspace)["observation"]["summary"]
                    if observation["status"] == "ready":
                        item.update(status="ready", sceneId=status["current"])
                        self.save(item)
                        return
                await asyncio.sleep(0.25)
            raise RuntimeError("Timed out waiting for the environment and initial camera images")
        except BaseException:
            await self.reclaim(item, "controller")
            raise

    def consume_events(self, item: dict, run_id: str):
        path = self.workspace(item["id"]) / "runs" / run_id / "events.jsonl"
        if not path.exists():
            return
        view = self.views[item["id"]]
        with path.open(encoding="utf-8") as stream:
            stream.seek(self.offsets.get(run_id, 0))
            while line := stream.readline():
                if not line.endswith("\n"):
                    break
                event = RunEvent.model_validate_json(line)
                view.accept(event)
                self.offsets[run_id] = stream.tell()

    async def send(self, item: dict, message: str, model: str | None):
        self.require_idle_operation(item["id"])
        if (
            item["status"] != "ready"
            or item.get("agent")
            or item["runStatus"] in {"running", "stopping"}
        ):
            raise ValueError("Wait for the scene and the previous run to be ready")
        process = self.controllers.get(item["id"])
        if process is None or process.returncode is not None:
            raise ValueError("Controller is not running")
        if not message.strip():
            raise ValueError("Instruction cannot be empty")
        workspace = self.workspace(item["id"])
        request = RunRequest(
            message=message,
            session_id=item["sessionId"],
            workspace=str(workspace),
            config=str(self.config_path) if self.config_path else None,
            model=model or item["model"],
            stream=True,
        )
        request_path = workspace.parent / "requests" / f"{request.run_id}.json"
        atomic_json(request_path, request.model_dump(mode="json"))
        item.update(runStatus="running", model=request.model, error=None)
        self.save(item)
        item["runs"].append(request.run_id)
        try:
            self.agents[item["id"]] = await self.spawn(
                item,
                "agent",
                [
                    sys.executable,
                    "-m",
                    "Emerge.cli.headless",
                    "--request",
                    str(request_path),
                    "--output-dir",
                    str(workspace / "runs" / request.run_id),
                ],
            )
        except Exception as exc:
            item.update(runStatus="failed", error=str(exc))
            self.save(item)
            raise
        return request.run_id

    async def stop(self, item: dict):
        process = self.agents.get(item["id"])
        if process and process.returncode is None:
            item["runStatus"] = "stopping"
            self.save(item)
            process.send_signal(signal.SIGINT)
            try:
                await asyncio.wait_for(asyncio.shield(process.wait()), timeout=20)
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    "Agent stop not confirmed; scene and workspace retained"
                ) from exc
        cancellation = await cancel_actions(
            self.workspace(item["id"]) / "ACTION.md", "web stop", 10
        )
        if not cancellation["acknowledged"]:
            item["runStatus"] = "stopping"
            self.save(item)
            raise RuntimeError("Robot action stop not confirmed; scene and workspace retained")
        if process:
            if item["runs"]:
                self.consume_events(item, item["runs"][-1])
            self.agents.pop(item["id"], None)
            item["agent"] = None
        if item["runStatus"] in {"running", "stopping"}:
            item["runStatus"] = "cancelled"
        if (
            item["status"] == "error"
            and self.controllers.get(item["id"])
            and self.controllers[item["id"]].returncode is None
        ):
            scene = read_json(self.workspace(item["id"]) / ".controller/scene.json")
            if scene and scene.get("ready"):
                item.update(status="ready", error=None)
        self.save(item)

    async def change_environment(self, item: dict, operation: str, scene_id: str | None):
        if operation == "switch" and scene_id not in {
            entry["id"] for entry in self.catalog["entries"]
        }:
            raise ValueError("Select a scene from the catalog")
        process = self.controllers.get(item["id"])
        if process is None or process.returncode is not None:
            raise ValueError("Restart the Controller before changing its environment")
        item.update(status="resetting" if operation == "reset" else "switching", error=None)
        self.save(item)
        await self.stop(item)
        workspace = self.workspace(item["id"])
        with WorkspaceLease(workspace):
            request = new_reset_request() if operation == "reset" else new_scene_request(scene_id)
            write_request(workspace, request)
            cleaned = False
            while True:
                if process.returncode is not None:
                    raise RuntimeError("Controller exited during the environment change")
                with control_lock(workspace):
                    result = read_json(result_file(workspace, request["request_id"]))
                    if result is None and expired(request):
                        result = write_result(workspace, request["request_id"], "expired")
                status = result["status"] if result else None
                if status in {"failed", "expired"}:
                    raise RuntimeError(
                        result.get("error", {}).get(
                            "message", "Controller did not accept the request"
                        )
                    )
                if status == "can_clean" and not cleaned:
                    try:
                        reset_workspace_context(workspace)
                    except Exception as exc:
                        update_request(
                            workspace,
                            request["request_id"],
                            "cleanup_failed",
                            {"code": "cleanup_failed", "message": str(exc)},
                        )
                        raise
                    item.update(sessionId="web:" + uuid4().hex, runs=[], runStatus="idle")
                    self.views[item["id"]] = RunView()
                    self.save(item)
                    cleaned = True
                    update_request(workspace, request["request_id"], "cleanup_completed")
                if status == "succeeded":
                    scene = read_json(workspace / ".controller/scene.json")
                    item.update(status="ready", sceneId=scene["current"], error=None)
                    self.save(item)
                    return
                await asyncio.sleep(0.25)

    async def close(self, item: dict, *, delete=False, archive=False):
        import shutil

        try:
            item.update(status="deleting" if delete else "closing", error=None)
            self.save(item)
            await self.stop(item)
            await self.reclaim(item, "agent")
            await self.reclaim(item, "controller")
            if delete:
                directory = self.workspace(item["id"]).parent
                if directory.exists():
                    shutil.rmtree(directory)
                self.db.execute("DELETE FROM instances WHERE id=?", (item["id"],))
                self.db.commit()
                del self.instances[item["id"]]
                self.views.pop(item["id"], None)
            else:
                item.update(status="closed", archived=archive)
                self.save(item)
        except Exception as exc:
            item.update(status="error", error=str(exc), operation=None)
            self.save(item)
            raise

    async def operate(self, item: dict, operation: str, scene_id: str | None = None):
        async with self.locks.setdefault(item["id"], asyncio.Lock()):
            if operation == "stop":
                await self.stop(item)
            elif operation in {"reset", "switch"}:
                await self.change_environment(item, operation, scene_id)
            elif operation == "restart":
                await self.launch(item)
            else:
                await self.close(item, delete=operation == "delete", archive=operation == "archive")

    def snapshot(self):
        output = []
        for item in list(self.instances.values()):
            key = item["id"]
            if key not in self.views:
                self.views[key] = RunView()
            if item["runs"]:
                for run_id in item["runs"]:
                    self.consume_events(item, run_id)
                run_id = item["runs"][-1]
                process = self.agents.get(key)
                if process and process.returncode is not None:
                    result = read_json(self.workspace(key) / "runs" / run_id / "result.json")
                    item.update(
                        agent=None,
                        runStatus=(
                            "stopping"
                            if item["operation"] == "stop"
                            else result["run_status"]
                            if result
                            else "failed"
                        ),
                    )
                    if result and result.get("error"):
                        item["error"] = result["error"]["message"]
                        if result["finish_reason"] == "cancellation_unconfirmed":
                            item["runStatus"] = "stopping"
                    if not result:
                        item["error"] = (
                            f"Agent exited ({process.returncode}) without a result; inspect its log."
                        )
                    self.agents.pop(key, None)
                    self.save(item)
            controller = self.controllers.get(key)
            if controller and controller.returncode is not None and item["status"] == "ready":
                item.update(
                    status="error",
                    error=f"Controller exited ({controller.returncode}); restart explicitly.",
                )
                self.save(item)
            snapshot = workspace_snapshot(self.workspace(key))
            view = self.views[key]
            messages = [
                message
                for message in view.messages
                if message["role"] != "assistant" or message["text"]
            ]
            observation = snapshot["observation"]
            manifest = self.workspace(key) / "artifacts/observations/observation.json"
            observation["updatedAt"] = manifest.stat().st_mtime_ns if manifest.exists() else None
            output.append(
                {
                    **item,
                    "workspace": str(self.workspace(key)),
                    "messages": messages,
                    "snapshot": snapshot,
                    "verification": view.verification,
                    "result": view.result,
                    "usage": view.usage,
                    "durationMs": view.duration_ms,
                }
            )
        return {"instances": output, "notice": self.notice}

    async def shutdown(self):
        self.closing = True
        for task in self.operations.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.operations.values(), return_exceptions=True)
        for item in list(self.instances.values()):
            try:
                await self.reclaim(item, "agent")
                await self.reclaim(item, "controller")
                item.update(
                    status="closed",
                    runStatus="cancelled" if item["runStatus"] == "running" else item["runStatus"],
                )
                self.save(item)
            except Exception as exc:
                item.update(status="error", error=str(exc))
                self.save(item)
        self.db.close()
        self.lease.__exit__(None, None, None)
