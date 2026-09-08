"""Prepare LIBERO-Pro's generated Environment dimension without third-party writes."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np
import torch


BASE_SUITE_ORDER = (
    "libero_goal",
    "libero_spatial",
    "libero_10",
    "libero_object",
)
NUM_INIT_STATES = 50
TARGET_ENVIRONMENT = "living_room_table"
MANIFEST_NAME = "manifest.json"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_cache_isolation(cache_root: Path, protected_roots: Iterable[Path]) -> None:
    resolved_cache = cache_root.resolve()
    for protected_root in protected_roots:
        resolved_protected = protected_root.resolve()
        if _is_relative_to(resolved_cache, resolved_protected):
            raise ValueError(
                "LIBERO-Pro generated-data cache must be outside third_party: "
                f"{resolved_cache} is inside {resolved_protected}"
            )


def _official_inputs(source_path: Path) -> tuple[Path, Path, Path]:
    perturbation_path = source_path / "perturbation.py"
    environment_config_path = source_path / "libero_ood/ood_environment.yaml"
    init_generator_path = source_path / "notebooks/generate_init_states.py"
    missing = [
        path
        for path in (
            perturbation_path,
            environment_config_path,
            init_generator_path,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "official LIBERO-Pro Environment generator is incomplete:\n"
            + "\n".join(str(path) for path in missing)
        )
    return perturbation_path, environment_config_path, init_generator_path


def _base_bddl_files(source_path: Path, suite: str) -> list[Path]:
    directory = source_path / "libero/libero/bddl_files" / suite
    files = sorted(directory.glob("*.bddl"))
    if len(files) != 10:
        raise ValueError(
            f"official LIBERO-Pro base suite {suite} has {len(files)} BDDLs; expected 10"
        )
    return files


def _input_digest(source_path: Path) -> str:
    perturbation_path, environment_config_path, init_generator_path = _official_inputs(
        source_path
    )
    hasher = hashlib.sha256()
    for path in (perturbation_path, environment_config_path, init_generator_path):
        hasher.update(path.relative_to(source_path).as_posix().encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    for suite in BASE_SUITE_ORDER:
        for path in _base_bddl_files(source_path, suite):
            hasher.update(path.relative_to(source_path).as_posix().encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(path.read_bytes())
            hasher.update(b"\0")
    hasher.update(f"num_init_states={NUM_INIT_STATES}\0".encode("ascii"))
    return hasher.hexdigest()


def _load_official_perturbation(path: Path) -> ModuleType:
    module_name = f"_libero_pro_perturbation_{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official LIBERO-Pro perturbation module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    return module


def _write_official_environment_bddls(
    source_path: Path,
    output_root: Path,
) -> dict[str, list[str]]:
    perturbation_path, environment_config_path, _ = _official_inputs(source_path)
    official = _load_official_perturbation(perturbation_path)
    written: dict[str, list[str]] = {}
    try:
        for suite in BASE_SUITE_ORDER:
            suite_name = f"{suite}_env"
            suite_output = output_root / "bddl_files" / suite_name
            suite_output.mkdir(parents=True, exist_ok=False)
            names: list[str] = []
            for source_bddl in _base_bddl_files(source_path, suite):
                content = source_bddl.read_text(encoding="utf-8")
                parser = official.BDDLParser(content)
                perturbator = official.EnvironmentReplacePerturbator(
                    parser,
                    str(environment_config_path),
                )
                generated = perturbator.perturb(
                    task_suite_name=suite,
                    task_name=source_bddl.stem,
                    seed=None,
                )
                (suite_output / source_bddl.name).write_text(
                    generated,
                    encoding="utf-8",
                )
                names.append(source_bddl.name)
            written[suite_name] = names
    finally:
        sys.modules.pop(official.__name__, None)
    return written


def _write_generation_config(config_dir: Path, source_path: Path, data_root: Path) -> None:
    benchmark_root = (source_path / "libero/libero").resolve()
    datasets = data_root / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_root": benchmark_root,
        "bddl_files": (data_root / "bddl_files").resolve(),
        "init_states": (data_root / "init_files").resolve(),
        "datasets": datasets.resolve(),
        "assets": (benchmark_root / "assets").resolve(),
    }
    (config_dir / "config.yaml").write_text(
        "".join(f"{key}: {value}\n" for key, value in paths.items()),
        encoding="utf-8",
    )


def _generate_official_init_states(source_path: Path, output_root: Path) -> None:
    _, _, init_generator_path = _official_inputs(source_path)
    compat_runner = Path(__file__).with_name("generate_init_states_compat.py")
    if not compat_runner.is_file():
        raise FileNotFoundError(f"missing LIBERO-Pro init compatibility runner: {compat_runner}")
    config_dir = output_root / ".libero_config"
    _write_generation_config(config_dir, source_path, output_root)
    runtime_env = os.environ.copy()
    runtime_env["LIBERO_CONFIG_PATH"] = str(config_dir.resolve())
    source_text = str(source_path.resolve())
    current_pythonpath = runtime_env.get("PYTHONPATH")
    runtime_env["PYTHONPATH"] = (
        source_text
        if not current_pythonpath
        else source_text + os.pathsep + current_pythonpath
    )
    runtime_env.setdefault("NUMBA_DISABLE_JIT", "1")
    runtime_env.setdefault("MUJOCO_GL", "egl")
    runtime_env["PYTHONDONTWRITEBYTECODE"] = "1"
    matplotlib_dir = output_root / ".matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    runtime_env.setdefault("MPLCONFIGDIR", str(matplotlib_dir.resolve()))

    for suite in BASE_SUITE_ORDER:
        suite_name = f"{suite}_env"
        bddl_dir = output_root / "bddl_files" / suite_name
        init_dir = output_root / "init_files" / suite_name
        init_dir.mkdir(parents=True, exist_ok=False)
        print(
            f"[LIBERO-Pro] Generating official Environment init states: {suite_name} "
            f"(10 tasks x {NUM_INIT_STATES})",
            flush=True,
        )
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(compat_runner),
                    str(init_generator_path),
                    "--bddl_base_dir",
                    str(bddl_dir),
                    "--output_dir",
                    str(init_dir),
                    "--num_inits",
                    str(NUM_INIT_STATES),
                ],
                cwd=output_root,
                env=runtime_env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"official LIBERO-Pro init generation failed for {suite_name} "
                f"with exit code {exc.returncode}"
            ) from exc

    for transient in (config_dir, output_root / "datasets", matplotlib_dir):
        if transient.exists():
            shutil.rmtree(transient)


def _load_init_states(path: Path) -> np.ndarray:
    try:
        states = torch.load(path)
    except Exception as exc:
        raise ValueError(f"cannot load generated init-state file {path}: {exc}") from exc
    return np.asarray(states)


def _artifact_digest(cache_dir: Path) -> str:
    hasher = hashlib.sha256()
    for kind, pattern in (("bddl_files", "*.bddl"), ("init_files", "*.pruned_init")):
        for path in sorted((cache_dir / kind).glob(f"*_env/{pattern}")):
            hasher.update(path.relative_to(cache_dir).as_posix().encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(path.read_bytes())
            hasher.update(b"\0")
    return hasher.hexdigest()


def _validate_cache(
    cache_dir: Path,
    source_path: Path,
    digest: str,
) -> dict[str, Any]:
    manifest_path = cache_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"missing generation manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid generation manifest {manifest_path}: {exc}") from exc
    if manifest.get("input_digest") != digest:
        raise ValueError(
            f"generated Environment cache digest does not match official inputs: {cache_dir}"
        )
    if manifest.get("num_init_states") != NUM_INIT_STATES:
        raise ValueError(
            f"generated Environment cache does not contain {NUM_INIT_STATES} states per task"
        )
    if manifest.get("artifact_digest") != _artifact_digest(cache_dir):
        raise ValueError(f"generated Environment cache content digest is invalid: {cache_dir}")

    for suite in BASE_SUITE_ORDER:
        suite_name = f"{suite}_env"
        expected_bddls = {path.name for path in _base_bddl_files(source_path, suite)}
        expected_inits = {f"{Path(name).stem}.pruned_init" for name in expected_bddls}
        bddl_dir = cache_dir / "bddl_files" / suite_name
        init_dir = cache_dir / "init_files" / suite_name
        actual_bddls = {path.name for path in bddl_dir.glob("*.bddl")}
        actual_inits = {path.name for path in init_dir.glob("*.pruned_init")}
        if actual_bddls != expected_bddls:
            raise ValueError(f"generated BDDL set is incomplete for {suite_name}")
        if actual_inits != expected_inits:
            raise ValueError(f"generated init-state set is incomplete for {suite_name}")
        for init_name in sorted(expected_inits):
            states = _load_init_states(init_dir / init_name)
            if states.ndim < 2 or len(states) != NUM_INIT_STATES:
                raise ValueError(
                    f"generated init-state shape is invalid for {init_dir / init_name}: "
                    f"{states.shape}"
                )
            if not np.isfinite(states).all():
                raise ValueError(f"generated init states are non-finite: {init_dir / init_name}")
    return manifest


def _write_manifest(
    output_root: Path,
    source_path: Path,
    digest: str,
    written: dict[str, list[str]],
) -> dict[str, Any]:
    manifest = {
        "schema_version": "Emerge.libero_pro_environment_cache.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_digest": digest,
        "artifact_digest": _artifact_digest(output_root),
        "generator": "official LIBERO-Pro perturbation.py + generate_init_states.py",
        "target_environment": TARGET_ENVIRONMENT,
        "num_init_states": NUM_INIT_STATES,
        "suites": written,
        "source_path": str(source_path.resolve()),
    }
    (output_root / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


@contextlib.contextmanager
def _cache_lock(cache_root: Path):
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / ".environment.lock"
    with lock_path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def prepare_environment_cache(
    source_path: Path,
    cache_root: Path,
    *,
    protected_roots: Iterable[Path] = (),
) -> tuple[Path, dict[str, Any]]:
    """Return a complete official Environment cache, generating it once if needed."""
    source_path = source_path.resolve()
    cache_root = cache_root.resolve()
    _require_cache_isolation(cache_root, (source_path, *protected_roots))
    digest = _input_digest(source_path)
    cache_dir = cache_root / f"environment-{digest[:16]}"

    with _cache_lock(cache_root):
        if cache_dir.exists():
            try:
                return cache_dir, _validate_cache(cache_dir, source_path, digest)
            except ValueError as exc:
                quarantine = cache_root / (
                    f".{cache_dir.name}.invalid-"
                    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                )
                print(
                    f"[LIBERO-Pro] Quarantining incomplete Environment cache: {exc}\n"
                    f"  {cache_dir} -> {quarantine}",
                    flush=True,
                )
                cache_dir.rename(quarantine)

        staging = Path(
            tempfile.mkdtemp(prefix=f".{cache_dir.name}.staging-", dir=cache_root)
        )
        try:
            print(
                "[LIBERO-Pro] Environment artifacts are absent; generating the official "
                f"40-task dimension in {cache_root}",
                flush=True,
            )
            written = _write_official_environment_bddls(source_path, staging)
            _generate_official_init_states(source_path, staging)
            manifest = _write_manifest(staging, source_path, digest, written)
            _validate_cache(staging, source_path, digest)
            staging.rename(cache_dir)
            return cache_dir, manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def _ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise ValueError(
                f"LIBERO-Pro data overlay has a stale link: {link} -> {link.resolve()} "
                f"(expected {target})"
            )
        return
    if link.exists():
        raise ValueError(f"refusing to replace existing overlay path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def prepare_data_overlay(
    output_dir: Path,
    frozen_data_path: Path,
    environment_cache: Path,
) -> Path:
    """Merge frozen data and generated Environment artifacts through symlinks."""
    overlay = output_dir / ".libero_pro/data"
    for kind in ("bddl_files", "init_files"):
        overlay_root = overlay / kind
        overlay_root.mkdir(parents=True, exist_ok=True)
        frozen_root = frozen_data_path / kind
        generated_root = environment_cache / kind
        generated_names = {source.name for source in generated_root.iterdir()}
        for source in sorted(frozen_root.iterdir()):
            if source.name in generated_names:
                continue
            _ensure_symlink(overlay_root / source.name, source)
        for source in sorted(generated_root.iterdir()):
            _ensure_symlink(overlay_root / source.name, source)
    return overlay
