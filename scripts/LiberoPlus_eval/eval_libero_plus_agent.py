#!/usr/bin/env python3
"""Run LIBERO-Plus robustness episodes with Emerge as the decision-making entrypoint.

LIBERO-Plus (https://github.com/sylvestf/LIBERO-plus) is a drop-in replacement for
LIBERO that expands the four standard suites into ~10,030 perturbed tasks spanning
seven robustness dimensions (Camera Viewpoints, Robot Initial States, Language
Instructions, Light Conditions, Background Textures, Sensor Noise, Objects Layout).

This driver reuses the entire episode lifecycle from ``eval_libero_agent`` (watchdog +
agent + libero_mujoco driver + selected VLA/WAM policy server, all unchanged). It adds the
things that are specific to LIBERO-Plus:

  1. It points ``libero_source_path`` at the LIBERO-Plus tree, so ``suite.get_task(id)``
     enumerates the thousands of perturbed tasks instead of the ten canonical ones.
  2. It selects an exact user-requested number of episodes per robustness dimension
     rather than selecting by suite or task id. Selection is deterministic for a
     seed and balanced round-robin across the four suites.
  3. It defaults to ``--trials-per-task 1`` (LIBERO-Plus official protocol; each
     perturbation is already its own task, so 50 trials would be 500k episodes).
  4. It aggregates per-episode results into the seven perturbation dimensions using
     ``benchmark/task_classification.json`` (task id -> category), and writes a
     dimension grid alongside the standard summary.

The policy backend is selected with ``--policy-backend``. WAM reuses the standard
LIBERO WAM embodiment profile. Nothing in the agent, controller, driver, policy
executors, or policy servers is modified.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# LIBERO-Plus imports robosuite while enumerating tasks. That import reaches
# numba-decorated robosuite functions whose source has no cache locator in this
# installation, so JIT must be disabled before even the shared evaluator is
# imported. Setting this unconditionally also prevents an inherited
# NUMBA_DISABLE_JIT=0 from reviving the import-time failure.
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Reuse the standard LIBERO evaluator wholesale. We only override enumeration and
# aggregation; the episode lifecycle (_run_episode, driver/agent/watchdog wiring,
# results.jsonl / summary.json writing) is shared verbatim.
REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_EVAL_DIR = REPO_ROOT / "scripts/Libero_eval"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LIBERO_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LIBERO_EVAL_DIR))

import eval_libero_agent as base  # noqa: E402

# LIBERO-Plus's env_wrapper.py imports `wand`, which dlopen's libMagickWand at import
# time. ImageMagick 7 lives in an isolated micromamba prefix
# (third_party/imagemagick_env/.env)
# so it never perturbs the EmergePolicy env. Point wand at it and prepend its lib dir to
# the loader path; both are inherited by the watchdog subprocess via os.environ.
IMAGEMAGICK_PREFIX = REPO_ROOT / "third_party/imagemagick_env/.env"


def _ensure_imagemagick_env() -> None:
    prefix = IMAGEMAGICK_PREFIX
    lib_dir = prefix / "lib"
    if not (lib_dir / "libMagickWand-7.Q16HDRI.so").exists():
        raise FileNotFoundError(
            f"ImageMagick prefix missing at {prefix}. LIBERO-Plus's env_wrapper needs "
            "libMagickWand for the Sensor Noise dimension. Create it with:\n"
            "  bash third_party/imagemagick_env/install.sh"
        )
    os.environ["MAGICK_HOME"] = str(prefix)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [str(lib_dir)] + ([existing] if existing else [])
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)

    # Preload wand now, before robosuite/mujoco pull in their own copies of shared
    # libs (libpng/libtiff/freetype/...). dlopen resolves symbols against whatever is
    # already loaded, so if robosuite loads first, libMagickWand later fails to bind.
    # Importing wand here pins ImageMagick's libs into the process first; subsequent
    # robosuite imports then can't displace them.
    import wand.image  # noqa: F401


def _install_robosuite_compat() -> None:
    """Install the 1.4->1.5 robosuite shim before any LIBERO import.

    LIBERO-Plus's envs/bddl_base_domain.py imports the legacy
    ``robosuite.environments.manipulation.single_arm_env`` symbol at module top
    level, so even --dry-run task enumeration (which imports libero.libero.benchmark)
    triggers it. The driver normally installs this shim when it builds an env
    (mujoco_env.py), but enumeration happens in-process here without a driver, so we
    invoke the same idempotent installer up front. Importing the installer also
    imports robosuite, so NUMBA_DISABLE_JIT is set at module startup rather than
    waiting for MujocoEnvManager.create(), which runs only after enumeration.
    """
    from robot.mujoco_simulation.mujoco_env import RobosuiteCompatibility

    RobosuiteCompatibility.install()


# LIBERO-Plus reuses the canonical suite names but with vastly more tasks per suite.
# task_classification.json only classifies the four manipulation suites below.
LIBERO_PLUS_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

# The seven robustness dimensions, in the column order used by the paper's grid and
# by dev/libero_plus_eval_grid.md.
DIMENSION_ORDER = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)


def _task_classification_path(source_path: Path) -> Path:
    return source_path / "libero/libero/benchmark/task_classification.json"


def _load_task_categories(source_path: Path) -> dict[str, dict[int, str]]:
    """Return {suite: {task_id0: category}} with task ids normalized to 0-based.

    task_classification.json numbers tasks from 1; the evaluator (and suite.get_task)
    use 0-based ids, so we subtract one on load.
    """
    path = _task_classification_path(source_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    categories: dict[str, dict[int, str]] = {}
    for suite, entries in raw.items():
        mapping: dict[int, str] = {}
        for entry in entries:
            mapping[int(entry["id"]) - 1] = str(entry["category"])
        categories[suite] = mapping
    return categories


def _parse_dimension(value: str) -> str:
    """Accept an official dimension name or its lowercase snake-case spelling."""
    normalized = "_".join(value.strip().lower().replace("-", " ").split())
    if normalized == "all":
        return "all"
    for dimension in DIMENSION_ORDER:
        if normalized == "_".join(dimension.lower().split()):
            return dimension
    choices = "all, " + ", ".join(
        "_".join(dimension.lower().split()) for dimension in DIMENSION_ORDER
    )
    raise argparse.ArgumentTypeError(
        f"unknown LIBERO-Plus dimension {value!r}; choose one of: {choices}"
    )


def _dimension_slug(dimension: str) -> str:
    return "_".join(dimension.lower().split())


def _dimension_rng(seed: int, dimension: str) -> random.Random:
    """Create a stable per-dimension RNG independent of CLI dimension order."""
    digest = hashlib.sha256(f"{seed}:{dimension}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big"))


def _select_task_ids_by_dimension(
    categories: dict[str, dict[int, str]],
    dimensions: list[str],
    *,
    episodes_per_dimension: int,
    seed: int,
) -> dict[str, dict[str, list[int]]]:
    """Select exactly N tasks per dimension, balanced round-robin by suite."""
    selected: dict[str, dict[str, list[int]]] = {}
    for dimension in dimensions:
        available = {
            suite: sorted(
                task_id
                for task_id, category in categories.get(suite, {}).items()
                if category == dimension
            )
            for suite in LIBERO_PLUS_SUITES
        }
        total_available = sum(len(task_ids) for task_ids in available.values())
        if episodes_per_dimension > total_available:
            raise ValueError(
                f"dimension {dimension!r} has {total_available} tasks, cannot select "
                f"{episodes_per_dimension}"
            )

        rng = _dimension_rng(seed, dimension)
        for task_ids in available.values():
            rng.shuffle(task_ids)

        positions = {suite: 0 for suite in LIBERO_PLUS_SUITES}
        chosen = {suite: [] for suite in LIBERO_PLUS_SUITES}
        chosen_count = 0
        while chosen_count < episodes_per_dimension:
            made_progress = False
            for suite in LIBERO_PLUS_SUITES:
                position = positions[suite]
                if position >= len(available[suite]):
                    continue
                chosen[suite].append(available[suite][position])
                positions[suite] += 1
                chosen_count += 1
                made_progress = True
                if chosen_count == episodes_per_dimension:
                    break
            if not made_progress:
                raise RuntimeError(
                    f"failed to allocate {episodes_per_dimension} tasks for {dimension}"
                )
        selected[dimension] = chosen
    return selected


def _round_robin_suite_specs(
    specs_by_suite: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Interleave suites deterministically while retaining every selected spec."""
    iterators = {
        suite: iter(specs_by_suite.get(suite, [])) for suite in LIBERO_PLUS_SUITES
    }
    active = [suite for suite in LIBERO_PLUS_SUITES if specs_by_suite.get(suite)]
    ordered: list[dict[str, Any]] = []
    while active:
        remaining: list[str] = []
        for suite in active:
            try:
                ordered.append(next(iterators[suite]))
            except StopIteration:
                continue
            remaining.append(suite)
        active = remaining
    return ordered


def _round_robin_dimension_specs(
    specs_by_dimension: dict[str, list[dict[str, Any]]],
    dimensions: list[str],
) -> list[dict[str, Any]]:
    """Interleave selected dimensions so partial runs cover each one evenly."""
    iterators = {dimension: iter(specs_by_dimension[dimension]) for dimension in dimensions}
    active = [dimension for dimension in dimensions if specs_by_dimension[dimension]]
    ordered: list[dict[str, Any]] = []
    while active:
        remaining: list[str] = []
        for dimension in active:
            try:
                ordered.append(next(iterators[dimension]))
            except StopIteration:
                continue
            remaining.append(dimension)
        active = remaining
    return ordered


def _build_dimension_specs(
    *,
    selected_task_ids: dict[str, dict[str, list[int]]],
    dimensions: list[str],
    start_trial: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build dimension-tagged episodes, balanced by both dimension and suite."""
    specs_by_dimension: dict[str, list[dict[str, Any]]] = {}
    for dimension in dimensions:
        specs_by_suite: dict[str, list[dict[str, Any]]] = {}
        slug = _dimension_slug(dimension)
        for suite in LIBERO_PLUS_SUITES:
            task_ids = selected_task_ids[dimension][suite]
            if not task_ids:
                continue
            # LIBERO-Plus's Benchmark._make_benchmark() prints the complete task
            # order (roughly 2,500 integers) every time a suite is constructed.
            # Dimension-first selection constructs several suites, so that
            # upstream debug print otherwise overwhelms the useful evaluator
            # progress. Exceptions still propagate; only construction stdout is
            # discarded.
            with contextlib.redirect_stdout(io.StringIO()):
                built = base._build_episode_specs(
                    suite_names=[suite],
                    task_ids_text=",".join(str(task_id) for task_id in task_ids),
                    trials_per_task=1,
                    start_trial=start_trial,
                    seed=seed,
                )
            by_task_id = {int(spec["task_id"]): spec for spec in built}
            tagged: list[dict[str, Any]] = []
            for task_id in task_ids:
                spec = dict(by_task_id[task_id])
                legacy_key = str(spec["key"])
                spec.update(
                    {
                        "dimension": dimension,
                        "dimension_slug": slug,
                        "legacy_key": legacy_key,
                        "key": f"{slug}:{legacy_key}",
                    }
                )
                tagged.append(spec)
            specs_by_suite[suite] = tagged
        specs_by_dimension[dimension] = _round_robin_suite_specs(specs_by_suite)
    return _round_robin_dimension_specs(specs_by_dimension, dimensions)


def _validate_bddl_sources(specs: list[dict[str, Any]]) -> None:
    """Preflight physical BDDL files, including LIBERO-Plus virtual names."""
    from robot.mujoco_simulation.mujoco_env import MujocoEnvManager

    missing: list[str] = []
    for spec in specs:
        requested = Path(spec["bddl_file"])
        source = MujocoEnvManager._bddl_source_file(requested)
        if source is None or not source.is_file():
            missing.append(f"{spec['key']}: {source or requested}")
    if missing:
        sample = "\n".join(missing[:8])
        remainder = len(missing) - 8
        suffix = f"\n... and {remainder} more" if remainder > 0 else ""
        raise FileNotFoundError(
            "LIBERO-Plus BDDL preflight failed before starting workers:\n"
            f"{sample}{suffix}"
        )


def _result_dimension(
    result: dict[str, Any],
    categories: dict[str, dict[int, str]],
) -> str | None:
    dimension = result.get("dimension")
    if isinstance(dimension, str) and dimension:
        return dimension
    suite = result.get("suite")
    task_id = result.get("task_id")
    if not isinstance(suite, str) or task_id is None:
        return None
    return categories.get(suite, {}).get(int(task_id))


def _completed_key_for_spec(
    spec: dict[str, Any],
    completed: dict[str, dict[str, Any]],
) -> str | None:
    """Match current dimension-first keys and pre-migration legacy keys."""
    for key in (spec["key"], spec.get("legacy_key")):
        if key and key in completed:
            return str(key)
    return None


def _write_evaluation_plan(
    output_dir: Path,
    *,
    dimensions: list[str],
    episodes_per_dimension: int,
    start_trial: int,
    seed: int,
    selected_task_ids: dict[str, dict[str, list[int]]],
    specs: list[dict[str, Any]],
    policy_backend: str,
    wam_conditioning_mode: str,
    driver_config_path: Path,
    profile_path: Path,
    resume: bool,
) -> None:
    """Persist the deterministic selection that makes resume reproducible."""
    stable_plan = {
        "dimensions": dimensions,
        "episodes_per_dimension": episodes_per_dimension,
        "total_episodes": len(specs),
        "start_trial": start_trial,
        "seed": seed,
        "episode_keys": [spec["key"] for spec in specs],
        "policy_backend": policy_backend,
        "wam_conditioning_mode": (
            wam_conditioning_mode if policy_backend == "wam" else None
        ),
        "profile_path": base._display_path(profile_path),
        "profile_sha256": base._sha256_file(profile_path),
        "driver_config_path": base._display_path(driver_config_path),
        "driver_config_sha256": base._sha256_file(driver_config_path),
    }
    plan_path = output_dir / "evaluation_plan.json"
    if resume and plan_path.exists():
        existing = base._load_json(plan_path)
        previous = {key: existing.get(key) for key in stable_plan}
        if previous != stable_plan:
            raise ValueError(
                "resume arguments, policy backend, or profile do not match "
                "evaluation_plan.json; use the original settings or choose a new "
                "output directory"
            )
    base._atomic_write_json(
        plan_path,
        {
            "schema_version": "Emerge.libero_plus_evaluation_plan.v2",
            "updated_at": base._utc_now(),
            **stable_plan,
        },
    )
    for dimension in dimensions:
        slug = _dimension_slug(dimension)
        task_ids_by_suite = selected_task_ids[dimension]
        base._atomic_write_json(
            output_dir / slug / "selection.json",
            {
                "schema_version": "Emerge.libero_plus_dimension_selection.v1",
                "dimension": dimension,
                "dimension_slug": slug,
                "episodes": sum(len(ids) for ids in task_ids_by_suite.values()),
                "seed": seed,
                "start_trial": start_trial,
                "suites": task_ids_by_suite,
            },
        )


def _write_dimension_summary(
    output_dir: Path,
    results: dict[str, dict[str, Any]],
    categories: dict[str, dict[int, str]],
) -> None:
    """Write a dimension-first root summary and one summary per dimension."""
    dimension_stats: dict[str, dict[str, Any]] = {}
    unclassified = {"episodes": 0, "successes": 0}
    for result in results.values():
        dimension = _result_dimension(result, categories)
        infrastructure_error = base._is_infrastructure_error(result)
        if dimension is None:
            if infrastructure_error:
                continue
            unclassified["episodes"] += 1
            unclassified["successes"] += int(bool(result.get("success")))
            continue
        dimension_item = dimension_stats.setdefault(
            dimension,
            {
                "episodes": 0,
                "successes": 0,
                "suites": {},
            },
        )
        suite = str(result.get("suite", "unknown"))
        suite_item = dimension_item["suites"].setdefault(
            suite, {"episodes": 0, "successes": 0}
        )
        if infrastructure_error:
            continue
        success = int(bool(result.get("success")))
        dimension_item["episodes"] += 1
        dimension_item["successes"] += success
        suite_item["episodes"] += 1
        suite_item["successes"] += success

    ordered_dimensions: dict[str, dict[str, Any]] = {}
    for dimension in DIMENSION_ORDER:
        if dimension not in dimension_stats:
            continue
        item = dimension_stats[dimension]
        item["success_rate"] = (
            item["successes"] / item["episodes"] if item["episodes"] else 0.0
        )
        for suite_item in item["suites"].values():
            suite_item["success_rate"] = (
                suite_item["successes"] / suite_item["episodes"]
                if suite_item["episodes"]
                else 0.0
            )
        ordered_dimensions[dimension] = item

    total_episodes = sum(item["episodes"] for item in ordered_dimensions.values())
    total_successes = sum(item["successes"] for item in ordered_dimensions.values())
    summary: dict[str, Any] = {
        "schema_version": "Emerge.libero_plus_evaluation_summary.v3",
        "updated_at": base._utc_now(),
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "dimensions": ordered_dimensions,
    }
    if unclassified["episodes"]:
        summary["unclassified"] = unclassified
    base._atomic_write_json(output_dir / "summary.json", summary)

    for dimension, item in ordered_dimensions.items():
        base._atomic_write_json(
            output_dir / _dimension_slug(dimension) / "summary.json",
            {
                "schema_version": "Emerge.libero_plus_dimension_summary.v2",
                "updated_at": summary["updated_at"],
                "dimension": dimension,
                **item,
            },
        )


def _write_dimension_grid(
    output_dir: Path,
    results: dict[str, dict[str, Any]],
    categories: dict[str, dict[int, str]],
) -> None:
    """Aggregate episode results into the seven-dimension robustness grid."""
    per_dim: dict[str, dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "successes": 0}
    )
    unclassified = {"episodes": 0, "successes": 0}
    for result in results.values():
        category = _result_dimension(result, categories)
        infrastructure_error = base._is_infrastructure_error(result)
        if infrastructure_error:
            continue
        success = int(bool(result.get("success")))
        if category is None:
            unclassified["episodes"] += 1
            unclassified["successes"] += success
            continue
        per_dim[category]["episodes"] += 1
        per_dim[category]["successes"] += success

    dimensions: dict[str, dict[str, Any]] = {}
    for dim in DIMENSION_ORDER:
        item = per_dim.get(dim, {"episodes": 0, "successes": 0})
        episodes = item["episodes"]
        dimensions[dim] = {
            "episodes": episodes,
            "successes": item["successes"],
            "success_rate": (item["successes"] / episodes) if episodes else 0.0,
        }

    total_episodes = sum(d["episodes"] for d in dimensions.values())
    total_successes = sum(d["successes"] for d in dimensions.values())
    grid = {
        "schema_version": "Emerge.libero_plus_dimension_grid.v2",
        "updated_at": base._utc_now(),
        "policy": "pi0.5",
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": (
            total_successes / total_episodes if total_episodes else 0.0
        ),
        "dimensions": dimensions,
    }
    if unclassified["episodes"]:
        grid["unclassified"] = unclassified
    base._atomic_write_json(output_dir / "libero_plus_grid.json", grid)

    # Human-readable one-liner table, aligned with dev/libero_plus_eval_grid.md columns.
    header = "| " + " | ".join(
        [d.split()[0] for d in DIMENSION_ORDER] + ["Total"]
    ) + " |"
    cells = [
        f"{dimensions[d]['success_rate'] * 100:.1f}" for d in DIMENSION_ORDER
    ]
    total_pct = (total_successes / total_episodes * 100) if total_episodes else 0.0
    cells.append(f"{total_pct:.2f}")
    row = "| " + " | ".join(cells) + " |"
    (output_dir / "libero_plus_grid.md").write_text(
        header + "\n" + row + "\n", encoding="utf-8"
    )


def main() -> int:
    _ensure_imagemagick_env()
    _install_robosuite_compat()
    parser = base._build_parser()
    # LIBERO-Plus environments are substantially heavier than standard LIBERO,
    # especially when several workers initialize MuJoCo and cameras together.
    parser.set_defaults(watchdog_ready_timeout_s=600.0)
    parser.description = (
        "Evaluate one or more LIBERO-Plus robustness dimensions with Emerge."
    )
    parser.add_argument(
        "--dimension",
        action="append",
        required=True,
        type=_parse_dimension,
        help=(
            "Robustness dimension to run; use the official quoted name, a "
            "snake-case name such as sensor_noise, or all. Repeat to select more "
            "than one."
        ),
    )
    parser.add_argument(
        "--episodes-per-dimension",
        "--count",
        dest="episodes_per_dimension",
        required=True,
        type=int,
        help=(
            "Exact number of episodes selected for each dimension. Total episodes "
            "equals this count multiplied by the number of unique dimensions."
        ),
    )
    # LIBERO-Plus is dimension-driven. Hide inherited suite/task selectors from
    # help and reject them below so there is only one source of task selection.
    for action in parser._actions:
        if action.dest in {"suite", "task_ids", "full"}:
            action.help = argparse.SUPPRESS
        elif action.dest == "policy_backend":
            action.choices = ("vla", "wam")
            action.default = "wam"
            action.help = (
                "Policy and profile used for LIBERO-Plus evaluation "
                "(default: wam)."
            )
    # Re-default trials to the LIBERO-Plus protocol (1 trial per perturbed task).
    # None lets us distinguish an explicitly supplied --task-ids from its inherited
    # standard-LIBERO default.
    parser.set_defaults(
        driver_config=REPO_ROOT / "dev/libero_plus_eval.json",
        task_ids=None,
    )
    args = parser.parse_args()

    if args.suite or args.task_ids is not None or args.full:
        parser.error(
            "LIBERO-Plus task selection is dimension-only; remove --suite, "
            "--task-ids, and --full. Suites are selected automatically and "
            "scheduled round-robin."
        )
    if args.episodes_per_dimension <= 0:
        parser.error("--episodes-per-dimension/--count must be positive")
    if args.trials_per_task != 1:
        parser.error(
            "LIBERO-Plus requires --trials-per-task 1 so total episodes remain "
            "count multiplied by the number of dimensions"
        )
    if args.start_trial < 0:
        parser.error("--start-trial cannot be negative")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        base._resolve_path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "artifacts/libero_plus_agent_eval" / timestamp
    )
    results_path = output_dir / "results.jsonl"
    if results_path.exists() and not args.resume:
        parser.error(
            f"{results_path} already exists; choose another output directory or use --resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    driver_config_path = base._resolve_path(args.driver_config)
    base_driver_config = base._load_json(driver_config_path)
    if not base_driver_config:
        parser.error(f"driver config is missing or invalid: {driver_config_path}")
    libero_config = base_driver_config.get("libero")
    if not isinstance(libero_config, dict):
        parser.error("driver config must contain a libero object")
    source_path = base._resolve_path(
        libero_config.get("libero_source_path", base.DEFAULT_LIBERO_SOURCE)
    )
    base._prepare_libero_config(output_dir, source_path)

    categories = _load_task_categories(source_path)

    requested_dimensions = set(args.dimension)
    dimensions = (
        list(DIMENSION_ORDER)
        if "all" in requested_dimensions
        else [
            dimension
            for dimension in DIMENSION_ORDER
            if dimension in requested_dimensions
        ]
    )
    try:
        selected_task_ids = _select_task_ids_by_dimension(
            categories,
            dimensions,
            episodes_per_dimension=args.episodes_per_dimension,
            seed=args.seed,
        )
        specs = _build_dimension_specs(
            selected_task_ids=selected_task_ids,
            dimensions=dimensions,
            start_trial=args.start_trial,
            seed=args.seed,
        )
        _validate_bddl_sources(specs)
        profile_path = base._policy_profile_path(
            base_driver_config,
            policy_backend=args.policy_backend,
            override=args.profile_path,
        )
        _write_evaluation_plan(
            output_dir,
            dimensions=dimensions,
            episodes_per_dimension=args.episodes_per_dimension,
            start_trial=args.start_trial,
            seed=args.seed,
            selected_task_ids=selected_task_ids,
            specs=specs,
            policy_backend=args.policy_backend,
            wam_conditioning_mode=args.wam_conditioning_mode,
            driver_config_path=driver_config_path,
            profile_path=profile_path,
            resume=args.resume,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"Prepared {len(specs)} LIBERO-Plus episode(s): "
        f"dimensions={dimensions}, episodes_per_dimension="
        f"{args.episodes_per_dimension}, workers={args.workers}, "
        "schedule=dimension-and-suite-round-robin"
    )
    for dimension in dimensions:
        suite_counts = ", ".join(
            f"{suite}={len(selected_task_ids[dimension][suite])}"
            for suite in LIBERO_PLUS_SUITES
        )
        print(f"  {dimension}: {suite_counts}")
    print(
        f"Policy: {args.policy_backend} | "
        f"Profile: {base._display_path(profile_path)} "
        f"(sha256={base._sha256_file(profile_path)[:12]}) | Output: {output_dir}"
    )
    if args.dry_run:
        for spec in specs:
            category = categories.get(spec["suite"], {}).get(spec["task_id"], "?")
            print(f"{spec['key']} | [{category}] | {spec['instruction']}")
        return 0

    if (
        args.policy_backend == "vla"
        and not args.skip_policy_server_check
        and not base._server_is_ready(base.OPENPI)
    ):
        parser.error(
            "VLA policy server is not reachable through discovery; "
            "start scripts/model_server/start_external_model_servers.sh --services openpi first or pass "
            "--skip-policy-server-check"
        )
    if (
        args.policy_backend == "wam"
        and not args.skip_wam_server_check
        and not base._server_is_ready(base.WAM_SERVICE)
    ):
        parser.error(
            "WAM policy server is not reachable through discovery; "
            "start external_model_server/cosmos_policy_server.py first or pass "
            "--skip-wam-server-check"
        )

    stored_results = base._read_results(results_path)
    completed: dict[str, dict[str, Any]] = {}
    pending_specs: list[tuple[int, dict[str, Any]]] = []
    for index, spec in enumerate(specs, start=1):
        if args.resume:
            completed_key = _completed_key_for_spec(spec, stored_results)
            if completed_key is not None:
                previous = stored_results[completed_key]
                if base._is_infrastructure_error(previous):
                    print(
                        f"[{index}/{len(specs)}] retry infrastructure error "
                        f"{spec['key']}"
                    )
                    pending_specs.append((index, spec))
                    continue
                completed[spec["key"]] = previous
                print(f"[{index}/{len(specs)}] skip completed {spec['key']}")
                continue
        pending_specs.append((index, spec))

    infrastructure_failed = False
    for _, spec, result in base._iter_episode_results(
        pending_specs,
        total_specs=len(specs),
        args=args,
        base_driver_config=base_driver_config,
        output_dir=output_dir,
        board=None,
    ):
        with results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        completed[spec["key"]] = result
        _write_dimension_summary(output_dir, completed, categories)
        _write_dimension_grid(output_dir, completed, categories)
        print(
            f"  success={result['success']} reason={result['termination_reason']} "
            f"steps={result['action_steps']} duration={result['duration_s']:.1f}s"
        )
        infrastructure_failed = (
            infrastructure_failed or base._is_infrastructure_error(result)
        )

    _write_dimension_summary(output_dir, completed, categories)
    _write_dimension_grid(output_dir, completed, categories)

    if infrastructure_failed and not args.continue_on_error:
        print(
            "Stopped scheduling new episodes after infrastructure error.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
