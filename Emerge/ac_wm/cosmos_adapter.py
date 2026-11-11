"""Cosmos3 framework adapter with auditable subprocess execution."""

import json
import os
import platform
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .protocol import ActionCandidate, RolloutRequest, RolloutResult


@dataclass(frozen=True, slots=True)
class CosmosFrameworkConfig:
    python: str
    checkpoint: str = "Cosmos3-Edge"
    config_file: str | None = None
    inference_module: str = "cosmos_framework.scripts.inference"
    parallelism_preset: str = "latency"
    seed: int = 0
    timeout_s: float = 1800.0
    server_url: str | None = None
    no_guardrails: bool = True
    action_horizon: int = 24
    libero_action_stats_path: str | None = None


def _cosmos_subprocess_environment(python: str) -> dict[str, str]:
    """Build a loader path from the selected Cosmos Python and host CUDA install."""
    env = os.environ.copy()
    cosmos_gpu = env.get("COSMOS_GPU")
    if cosmos_gpu:
        env["CUDA_VISIBLE_DEVICES"] = cosmos_gpu

    prefix = Path(python).resolve().parent.parent
    library_dirs = [prefix / "lib"]
    for site_packages in sorted((prefix / "lib").glob("python*/site-packages")):
        library_dirs.extend(
            site_packages / relative
            for relative in ("av.libs", "opencv_python.libs", "nvidia/npp/lib")
        )

    cuda_home = Path(env.get("CUDA_HOME", "/usr/local/cuda"))
    library_dirs.extend(
        (cuda_home / "targets" / f"{platform.machine()}-linux" / "lib", cuda_home / "lib64")
    )
    existing = [path for path in env.get("LD_LIBRARY_PATH", "").split(":") if path]
    paths = [str(path) for path in library_dirs if path.exists()] + existing
    env["LD_LIBRARY_PATH"] = ":".join(paths)
    return env


# Emerge 7D LIBERO rows -> Cosmos LIBERO 10D frame-wise-relative input.
_OSC7_TRANSFORM = "osc7_controls_to_libero10_framewise_relative_rot6d_gripper01_open1_quantile_rot"
_SHORT_HORIZON_ERROR = "candidate horizon is shorter than Cosmos minimum of 4 rows"


def _action_transform(candidate: ActionCandidate) -> str:
    if len(candidate.actions[0]) == 7:
        return _OSC7_TRANSFORM
    return str(candidate.metadata.get("action_representation", "libero10_rot6d_gripper01"))


def _run_cosmos_process(
    command, *, cwd: str, env: dict[str, str], timeout_s: float
) -> subprocess.CompletedProcess:
    """Run Cosmos in its own process group so a timeout also stops its worker processes."""
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(command, timeout_s, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


class CosmosFrameworkAdapter:
    def __init__(self, config: CosmosFrameworkConfig) -> None:
        self.config = config

    def write_input_spec(
        self, request: RolloutRequest, candidate: ActionCandidate, path: str
    ) -> str:
        if request.domain_name != "libero":
            raise ValueError(f"unsupported action domain: {request.domain_name}")
        width = len(candidate.actions[0])
        if width not in (7, 10):
            raise ValueError(f"LIBERO action width must be 7 or 10, got {width}")
        out = Path(path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.config.action_horizon < 4 or self.config.action_horizon % 4:
            raise ValueError("action_horizon must be a positive multiple of 4 and at least 4")
        raw_actions = tuple(tuple(float(v) for v in row) for row in candidate.actions)
        controller_rows = np.asarray(raw_actions, dtype=np.float64)
        if width == 7:
            # Cosmos LIBERO training data stores the environment's normalized OSC
            # controls verbatim: translation stays in controller units and only the
            # axis-angle control is re-encoded as rot6d (official
            # LIBEROLeRobotDataset._build_frame_wise_action). The model's gripper is
            # [0, 1] with 1=open, while LIBERO uses -1=open/+1=close; the official
            # closed-loop client maps back with env = 1 - 2g.
            controller_rows = np.clip(controller_rows, -1.0, 1.0)
            matrices = Rotation.from_rotvec(controller_rows[:, 3:6]).as_matrix()
            rot6d = matrices[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
            gripper01 = (1.0 - controller_rows[:, 6:7]) * 0.5
            frame_actions = np.concatenate([controller_rows[:, :3], rot6d, gripper01], axis=1)
        else:
            frame_actions = controller_rows
        stats_path = self.config.libero_action_stats_path or ""
        if not stats_path:
            raise ValueError(
                "libero_action_stats_path is required for Cosmos LIBERO quantile_rot input"
            )
        stats = json.loads(Path(stats_path).read_text())
        raw_stats = stats.get("global_raw", stats.get("global", {}))
        q01 = np.asarray(raw_stats["q01"], dtype=np.float64)
        q99 = np.asarray(raw_stats["q99"], dtype=np.float64)
        if q01.shape != (10,) or q99.shape != (10,):
            raise ValueError(
                f"LIBERO quantile stats must have width 10, got {q01.shape}/{q99.shape}"
            )
        model_actions = 2.0 * (frame_actions - q01) / np.maximum(q99 - q01, 1e-8) - 1.0
        requested_prediction_steps = max(
            1,
            min(
                int(candidate.metadata.get("prediction_steps", len(model_actions))),
                len(model_actions),
            ),
        )
        horizon = min(requested_prediction_steps, self.config.action_horizon)
        horizon -= horizon % 4
        if horizon < 4:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} has only {requested_prediction_steps} prediction rows; Cosmos action forward dynamics requires at least 4"
            )
        actions = model_actions[:horizon]
        action_path = out.with_suffix(".actions.json")
        action_path.write_text(json.dumps(actions.tolist(), indent=2) + "\n")
        # Cosmos was trained on single-sentence task captions; drop the planner's
        # appended comparison instructions, which are meant for the VLM judge.
        prompt = (request.task.strip().splitlines() or [""])[0].strip()
        if request.view_point == "concat_view":
            prompt += " The left half shows the third-person view; the right half shows the wrist-mounted camera."
        payload = {
            "model_mode": "forward_dynamics",
            "name": candidate.candidate_id,
            "domain_name": request.domain_name,
            "vision_path": str(Path(request.observation_path).resolve()),
            "action_path": str(action_path),
            "action_chunk_size": horizon,
            "num_frames": horizon + 1,
            "prompt": prompt,
            "fps": 20,
            "image_size": 256,
        }
        if request.view_point is not None:
            payload["view_point"] = request.view_point
        out.write_text(json.dumps(payload, indent=2) + "\n")
        return str(out)

    def command(self, input_spec: str, output_dir: str) -> tuple[str, ...]:
        if not self.config.python:
            raise ValueError("COSMOS_PYTHON is required when using the Cosmos CLI backend")
        if not self.config.checkpoint:
            raise ValueError("COSMOS_CHECKPOINT is required when using the Cosmos CLI backend")
        args = [self.config.python, "-m", self.config.inference_module]
        if self.config.no_guardrails:
            args.append("--no-guardrails")
        if self.config.config_file:
            args += ["--config-file", self.config.config_file]
        args += [
            "--parallelism-preset",
            self.config.parallelism_preset,
            "-i",
            input_spec,
            "-o",
            output_dir,
            "--checkpoint-path",
            self.config.checkpoint,
            "--seed",
            str(self.config.seed),
        ]
        return tuple(args)

    def _transform_metadata(self, request: RolloutRequest, candidate: ActionCandidate) -> dict:
        return {
            "action_transform": _action_transform(candidate),
            "action_stats_path": self.config.libero_action_stats_path or "",
            "model_action_width": 10,
            "view_point": request.view_point or "ego_view",
        }

    def run(
        self, request: RolloutRequest, candidate: ActionCandidate, work_dir: str
    ) -> RolloutResult:
        if self.config.server_url:
            return self.run_many(request, (candidate,), work_dir)[candidate.candidate_id]
        root = Path(work_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        spec = self.write_input_spec(
            request, candidate, str(root / f"{candidate.candidate_id}.json")
        )
        spec_payload = json.loads(Path(spec).read_text())
        horizon = int(spec_payload["action_chunk_size"])
        requested_steps = max(
            1,
            min(
                int(candidate.metadata.get("execute_steps", len(candidate.actions))),
                len(candidate.actions),
            ),
        )
        output = root / candidate.candidate_id
        output.mkdir(exist_ok=True)
        env = _cosmos_subprocess_environment(self.config.python)
        try:
            p = _run_cosmos_process(
                self.command(spec, str(output)),
                cwd=str(root),
                env=env,
                timeout_s=self.config.timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            (root / f"{candidate.candidate_id}.stderr").write_text(
                (e.stderr or "") if isinstance(e.stderr, str) else ""
            )
            return RolloutResult(
                candidate.candidate_id,
                "timeout",
                metadata={"output_dir": str(output)},
                error=f"timeout after {self.config.timeout_s}s",
            )
        (root / f"{candidate.candidate_id}.stdout").write_text(p.stdout)
        (root / f"{candidate.candidate_id}.stderr").write_text(p.stderr)
        if p.returncode != 0:
            return RolloutResult(
                candidate.candidate_id,
                "failed",
                metadata={"returncode": p.returncode, "output_dir": str(output)},
                error=p.stderr[-4000:],
            )
        videos = list(output.rglob("vision.mp4"))
        return RolloutResult(
            candidate.candidate_id,
            "success" if videos else "failed",
            str(videos[0]) if videos else None,
            {
                "returncode": 0,
                "inference_backend": "cosmos_cli",
                "output_dir": str(output),
                "raw_action_chunk_size": len(candidate.actions),
                "requested_execute_steps": requested_steps,
                "requested_prediction_steps": int(
                    candidate.metadata.get("prediction_steps", len(candidate.actions))
                ),
                "action_horizon": horizon,
                "num_frames": horizon + 1,
                "action_sequence_policy": "real_proposal_rows_only_no_padding",
                **self._transform_metadata(request, candidate),
            },
            None if videos else "missing vision.mp4",
        )

    def _run_many_via_service(self, request, candidates, root, output, payloads):
        """Call one persistent Cosmos pipeline; fail closed when the configured service is unavailable."""
        from uuid import uuid4

        from websockets.sync.client import connect

        from external_model_server.ac_wm_contract import EXPECTATION, INPUT_SCHEMA
        from external_model_server.model_service.codec import pack_message, unpack_message
        from external_model_server.model_service.contracts import (
            PROTOCOL_VERSION,
            response_result,
            validate_description,
        )

        started = time.monotonic()
        timeout = self.config.timeout_s
        message = {
            "output_dir": str(output),
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "input_spec": str(root / f"{candidate.candidate_id}.json"),
                }
                for candidate in candidates
            ],
        }
        request_id = uuid4().hex
        with connect(
            self.config.server_url,
            open_timeout=10,
            close_timeout=5,
            proxy=None,
            ping_interval=20,
            ping_timeout=None,
            max_size=None,
        ) as websocket:
            hello = validate_description(unpack_message(websocket.recv(timeout=10)))
            EXPECTATION.check(hello)
            websocket.send(
                pack_message(
                    {
                        "request_id": request_id,
                        "protocol_version": PROTOCOL_VERSION,
                        "operation": "infer",
                        "input_schema": INPUT_SCHEMA,
                        "timeout_s": timeout,
                        "payload": message,
                    }
                )
            )
            body = response_result(unpack_message(websocket.recv(timeout=timeout)), request_id)
        elapsed = time.monotonic() - started
        by_id = {item.get("candidate_id"): item for item in body.get("results", [])}
        results = {}
        for candidate in candidates:
            item = by_id.get(candidate.candidate_id)
            metadata = {
                "service_url": self.config.server_url,
                "inference_backend": "persistent_cosmos_service",
                "service_request_count": body.get("request_count"),
                "service_batch_elapsed_s": round(elapsed, 3),
                "output_dir": str(output / candidate.candidate_id),
                "input_spec": str(root / f"{candidate.candidate_id}.json"),
            }
            if item is None:
                results[candidate.candidate_id] = RolloutResult(
                    candidate.candidate_id,
                    "failed",
                    metadata=metadata,
                    error="Cosmos service omitted candidate result",
                )
                continue
            metadata["service_sample_status"] = item.get("sample_status")
            results[candidate.candidate_id] = RolloutResult(
                candidate.candidate_id,
                item.get("status", "failed"),
                item.get("video_path"),
                metadata,
                item.get("error"),
            )
        return results

    def _run_many_via_cli(self, candidates, root, output, batch_file):
        env = _cosmos_subprocess_environment(self.config.python)
        try:
            proc = _run_cosmos_process(
                self.command(str(batch_file), str(output)),
                cwd=str(root),
                env=env,
                timeout_s=self.config.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            (root / "cosmos_batch.stderr").write_text(stderr)
            return {
                c.candidate_id: RolloutResult(
                    c.candidate_id,
                    "timeout",
                    metadata={
                        "inference_backend": "cosmos_cli",
                        "output_dir": str(output / c.candidate_id),
                    },
                    error=f"timeout after {self.config.timeout_s}s",
                )
                for c in candidates
            }
        (root / "cosmos_batch.stdout").write_text(proc.stdout)
        (root / "cosmos_batch.stderr").write_text(proc.stderr)
        results = {}
        for candidate in candidates:
            candidate_output = output / candidate.candidate_id
            videos = list(candidate_output.rglob("vision.mp4"))
            metadata = {
                "inference_backend": "cosmos_cli",
                "returncode": proc.returncode,
                "output_dir": str(candidate_output),
                "batch_output_dir": str(output),
            }
            error = (
                proc.stderr[-4000:]
                if proc.returncode
                else (None if videos else "missing vision.mp4")
            )
            ok = proc.returncode == 0 and bool(videos)
            results[candidate.candidate_id] = RolloutResult(
                candidate.candidate_id,
                "success" if ok else "failed",
                str(videos[0]) if ok else None,
                metadata,
                error,
            )
        return results

    def run_many(
        self, request: RolloutRequest, candidates: tuple[ActionCandidate, ...], work_dir: str
    ) -> dict[str, RolloutResult]:
        """Run a shared-state candidate batch through a resident service or CLI backend.

        Always returns one result per requested candidate. A candidate too short
        for Cosmos fails individually instead of failing the whole batch.
        """
        root = Path(work_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        requested_execute_by_id = {
            candidate.candidate_id: max(
                1,
                min(
                    int(candidate.metadata.get("execute_steps", len(candidate.actions))),
                    len(candidate.actions),
                ),
            )
            for candidate in candidates
        }
        requested_prediction_by_id = {
            candidate.candidate_id: max(
                1,
                min(
                    int(candidate.metadata.get("prediction_steps", len(candidate.actions))),
                    len(candidate.actions),
                ),
            )
            for candidate in candidates
        }
        horizons_by_id = {}
        payloads = {}
        for candidate in candidates:
            # Preserve each skill's own real proposal prefix. Never shorten a long
            # candidate merely because another skill proposed fewer executable rows.
            horizon = min(
                self.config.action_horizon, requested_prediction_by_id[candidate.candidate_id]
            )
            horizon -= horizon % 4
            horizons_by_id[candidate.candidate_id] = horizon
            if horizon < 4:
                continue
            prediction_candidate = ActionCandidate(
                candidate.candidate_id,
                candidate.skill_name,
                candidate.actions,
                {**dict(candidate.metadata), "prediction_steps": horizon},
            )
            spec = self.write_input_spec(
                request, prediction_candidate, str(root / f"{candidate.candidate_id}.json")
            )
            payloads[candidate.candidate_id] = json.loads(Path(spec).read_text())
        eligible = tuple(c for c in candidates if c.candidate_id in payloads)
        batch_file = root / "cosmos_candidates.jsonl"
        output = root / "batch_outputs"
        backend_results = {}
        if eligible:
            batch_file.write_text(
                "".join(json.dumps(payloads[c.candidate_id]) + "\n" for c in eligible)
            )
            output.mkdir(exist_ok=True)
            if self.config.server_url:
                backend_results = self._run_many_via_service(
                    request, eligible, root, output, payloads
                )
            else:
                backend_results = self._run_many_via_cli(eligible, root, output, batch_file)
        results = {}
        for candidate in candidates:
            cid = candidate.candidate_id
            metadata = {
                "raw_action_chunk_size": len(candidate.actions),
                "requested_execute_steps": requested_execute_by_id[cid],
                "requested_prediction_steps": requested_prediction_by_id[cid],
                "candidate_prediction_horizon": horizons_by_id[cid],
                "action_sequence_policy": "candidate_real_proposal_prefix_no_padding",
                **self._transform_metadata(request, candidate),
            }
            if cid not in payloads:
                results[cid] = RolloutResult(
                    cid, "failed", metadata=metadata, error=_SHORT_HORIZON_ERROR
                )
                continue
            horizon = int(payloads[cid]["action_chunk_size"])
            metadata.update(
                {
                    "batch_input": str(batch_file),
                    "action_horizon": horizon,
                    "num_frames": horizon + 1,
                    "comparison_policy": "candidate_native_horizon_no_padding; judge only credits each candidate's dispatched prefix",
                }
            )
            backend = backend_results.get(cid)
            if backend is None:
                results[cid] = RolloutResult(
                    cid,
                    "failed",
                    metadata=metadata,
                    error="Cosmos backend omitted candidate result",
                )
                continue
            metadata.update(backend.metadata)
            results[cid] = RolloutResult(
                cid, backend.status, backend.video_path, metadata, backend.error
            )
        return results

    def failed(self, candidate: ActionCandidate, error: str) -> RolloutResult:
        return RolloutResult(candidate.candidate_id, "failed", error=error)
