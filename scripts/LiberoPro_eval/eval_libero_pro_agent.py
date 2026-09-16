#!/usr/bin/env python3
"""Run dimension-first LIBERO-Pro evaluations with Emerge.

This entrypoint is intentionally self-contained. It reuses the standard LIBERO
episode lifecycle without modifying either the standard LIBERO evaluator or the
LIBERO-Plus evaluator.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# LIBERO imports robosuite modules decorated by numba. Disable JIT before the
# shared evaluator imports the controller / robosuite so Python 3.12 can enumerate tasks.
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_EVAL_DIR = REPO_ROOT / "scripts/Libero_eval"
LIBERO_PRO_EVAL_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LIBERO_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LIBERO_EVAL_DIR))
if str(LIBERO_PRO_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LIBERO_PRO_EVAL_DIR))

import eval_libero_agent as base  # noqa: E402
from environment_dimension import (  # noqa: E402
    prepare_data_overlay,
    prepare_environment_cache,
)


BASE_SUITE_ORDER = (
    "libero_goal",
    "libero_spatial",
    "libero_10",
    "libero_object",
)
DIMENSION_ORDER = ("Object", "Position", "Semantic", "Task", "Environment")
DIMENSION_SUFFIX = {
    "Object": "object",
    "Position": "swap",
    "Semantic": "lan",
    "Task": "task",
    "Environment": "env",
}
DIMENSION_ALIASES = {
    "object": "Object",
    "position": "Position",
    "swap": "Position",
    "semantic": "Semantic",
    "language": "Semantic",
    "lan": "Semantic",
    "task": "Task",
    "environment": "Environment",
    "env": "Environment",
}
LANGUAGE_PATTERN = re.compile(
    r"^\s*\(:language\s+(.+?)\s*\)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
MAX_TASKS_PER_DIMENSION = len(BASE_SUITE_ORDER) * 10

# Scope exclusions by derived suite and BDDL filename. The same base filename
# can be valid in another perturbation dimension.
EXCLUDED_BDDLS: dict[str, dict[str, str]] = {
    "libero_spatial_swap": {
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl": (
            "user-requested stability exclusion: the elevated bowl-on-cookie-box "
            "grasp is unreliable, although a targeted retry has succeeded"
        ),
    },
    "libero_10_object": {
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl": (
            "Object perturbation requires manipulating bigger_alphabet_soup, "
            "whose oversized body cannot be reliably enclosed by the Panda gripper"
        ),
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl": (
            "Object perturbation requires manipulating bigger_alphabet_soup, "
            "whose oversized body cannot be reliably enclosed by the Panda gripper"
        ),
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl": (
            "instruction asks for a book, but the scene has no book and the goal "
            "requires black_bowl_1"
        ),
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl": (
            "instruction identifies a yellow-and-white mug, but the goal requires "
            "red_coffee_mug_1 while another mug is present"
        ),
    },
    "libero_object_object": {
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl": (
            "Object perturbation replaces alphabet soup with bigger_alphabet_soup, "
            "whose oversized body cannot be reliably enclosed by the Panda gripper"
        ),
    },
    "libero_10_env": {
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl": (
            "Environment replacement is a no-op because the source workspace is "
            "already living_room_table"
        ),
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.bddl": (
            "Environment replacement is a no-op because the source workspace is "
            "already living_room_table"
        ),
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl": (
            "Environment replacement is a no-op because the source workspace is "
            "already living_room_table"
        ),
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl": (
            "Environment replacement is a no-op because the source workspace is "
            "already living_room_table"
        ),
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl": (
            "Environment replacement is a no-op because the source workspace is "
            "already living_room_table"
        ),
    },
}


def _install_robosuite_compat() -> None:
    """Install the existing Emerge robosuite 1.4 -> 1.5 shim."""
    from robot.mujoco_simulation.mujoco_env import RobosuiteCompatibility

    RobosuiteCompatibility.install()


def _parse_dimension(value: str) -> str:
    normalized = "_".join(value.strip().lower().replace("-", " ").split())
    if normalized in {"all", "available"}:
        return normalized
    dimension = DIMENSION_ALIASES.get(normalized)
    if dimension is not None:
        return dimension
    choices = "all, available, object, position, semantic, task, environment"
    raise argparse.ArgumentTypeError(
        f"unknown LIBERO-Pro dimension {value!r}; choose one of: {choices}"
    )


def _parse_only_task(value: str) -> tuple[str, int]:
    suite_name, separator, task_id_text = value.strip().rpartition(":")
    if not separator or not suite_name or not task_id_text:
        raise argparse.ArgumentTypeError(
            "--only-task must use DERIVED_SUITE:TASK_ID, for example "
            "libero_spatial_swap:3"
        )
    try:
        task_id = int(task_id_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--only-task task ID must be an integer: {value!r}"
        ) from exc
    if task_id < 0:
        raise argparse.ArgumentTypeError(
            f"--only-task task ID cannot be negative: {value!r}"
        )
    return suite_name, task_id


def _dimension_slug(dimension: str) -> str:
    return dimension.lower()


def _benchmark_suite(base_suite: str, dimension: str) -> str:
    return f"{base_suite}_{DIMENSION_SUFFIX[dimension]}"


def _dimension_rng(seed: int, dimension: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{dimension}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big"))


def _dimension_missing_paths(data_path: Path, dimension: str) -> list[Path]:
    missing: list[Path] = []
    for base_suite in BASE_SUITE_ORDER:
        suite = _benchmark_suite(base_suite, dimension)
        for directory in (
            data_path / "bddl_files" / suite,
            data_path / "init_files" / suite,
        ):
            if not directory.is_dir():
                missing.append(directory)
    return missing


def _available_dimensions(data_path: Path) -> list[str]:
    return [
        dimension
        for dimension in DIMENSION_ORDER
        if not _dimension_missing_paths(data_path, dimension)
    ]


def _resolve_dimensions(requested: list[str], data_path: Path) -> list[str]:
    requested_set = set(requested)
    available = _available_dimensions(data_path)
    if "all" in requested_set:
        dimensions = list(DIMENSION_ORDER)
    else:
        selected = {
            item
            for item in requested_set
            if item not in {"all", "available"}
        }
        if "available" in requested_set:
            selected.update(available)
        dimensions = [item for item in DIMENSION_ORDER if item in selected]
    if not dimensions:
        raise ValueError("at least one available LIBERO-Pro dimension is required")

    missing_by_dimension = {
        dimension: _dimension_missing_paths(data_path, dimension)
        for dimension in dimensions
    }
    missing_by_dimension = {
        dimension: paths
        for dimension, paths in missing_by_dimension.items()
        if paths
    }
    if missing_by_dimension:
        lines = []
        for dimension, paths in missing_by_dimension.items():
            sample = ", ".join(str(path) for path in paths[:2])
            lines.append(f"{dimension}: {sample}")
        detail = "\n".join(lines)
        raise FileNotFoundError(
            "requested LIBERO-Pro dimension data is incomplete:\n"
            f"{detail}\n"
            "Use --dimension available to run only locally complete dimensions."
        )
    return dimensions


def _requests_environment(requested: list[str]) -> bool:
    return bool({"Environment", "available", "all"}.intersection(requested))


def _prepare_libero_pro_config(
    output_dir: Path,
    source_path: Path,
    data_path: Path,
) -> Path:
    """Create a run-local LIBERO config with code and data kept separate."""
    benchmark_root = source_path / "libero/libero"
    paths = {
        "benchmark_root": benchmark_root,
        "bddl_files": data_path / "bddl_files",
        "init_states": data_path / "init_files",
        "datasets": data_path / "datasets",
        "assets": benchmark_root / "assets",
    }
    required_names = ("benchmark_root", "bddl_files", "init_states", "assets")
    missing = [
        f"{name}: {paths[name]}"
        for name in required_names
        if not paths[name].exists()
    ]
    if missing:
        raise FileNotFoundError(
            "LIBERO-Pro code/data checkout is incomplete:\n" + "\n".join(missing)
        )

    config_dir = output_dir / ".libero_pro"
    config_dir.mkdir(parents=True, exist_ok=True)
    if not paths["datasets"].exists():
        paths["datasets"] = config_dir / "datasets"
        paths["datasets"].mkdir(parents=True, exist_ok=True)
    config_text = "".join(f"{key}: {value}\n" for key, value in paths.items())
    (config_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    loaded = sys.modules.get("libero.libero")
    loaded_file = Path(getattr(loaded, "__file__", "")).resolve() if loaded else None
    if loaded_file is not None and source_path.resolve() not in loaded_file.parents:
        raise RuntimeError(
            "another LIBERO checkout was imported before LIBERO-Pro: "
            f"{loaded_file}"
        )
    source_text = str(source_path)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return config_dir


def _load_benchmark_dict(source_path: Path) -> dict[str, Any]:
    benchmark, _ = base._load_benchmark_api()
    package = sys.modules.get("libero.libero")
    package_file = Path(getattr(package, "__file__", "")).resolve() if package else None
    if package_file is None or source_path.resolve() not in package_file.parents:
        raise RuntimeError(
            "LIBERO-Pro source isolation failed; imported package is "
            f"{package_file or 'unknown'}"
        )
    return benchmark.get_benchmark_dict()


def _construct_suite(benchmark_dict: dict[str, Any], suite_name: str) -> Any:
    suite_class = benchmark_dict.get(suite_name)
    if suite_class is None:
        raise ValueError(f"LIBERO-Pro suite is not registered: {suite_name}")
    with contextlib.redirect_stdout(io.StringIO()):
        suite = suite_class()
    if int(suite.get_num_tasks()) != 10:
        raise ValueError(
            f"LIBERO-Pro suite {suite_name} has {suite.get_num_tasks()} tasks; expected 10"
        )
    return suite


def _partition_suite_task_ids(
    suite: Any,
    suite_name: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Return eligible IDs and the audited exclusions for one derived suite."""
    policy = EXCLUDED_BDDLS.get(suite_name, {})
    eligible: list[int] = []
    excluded: list[dict[str, Any]] = []
    seen_bddls: set[str] = set()
    for task_id in range(int(suite.get_num_tasks())):
        task = suite.get_task(task_id)
        bddl_file = Path(str(task.bddl_file)).name
        seen_bddls.add(bddl_file)
        reason = policy.get(bddl_file)
        if reason is None:
            eligible.append(task_id)
            continue
        excluded.append(
            {
                "task_id": task_id,
                "task_name": str(task.name),
                "bddl_file": bddl_file,
                "reason": reason,
            }
        )

    missing = sorted(set(policy) - seen_bddls)
    if missing:
        raise ValueError(
            f"LIBERO-Pro exclusion policy is stale for {suite_name}; "
            f"missing BDDL(s): {', '.join(missing)}"
        )
    return eligible, excluded


def _select_task_ids_by_dimension(
    benchmark_dict: dict[str, Any],
    dimensions: list[str],
    *,
    tasks_per_dimension: int,
    seed: int,
) -> tuple[
    dict[str, dict[str, list[int]]],
    dict[str, dict[str, list[dict[str, Any]]]],
]:
    """Select up to N eligible tasks per dimension, round-robin by base suite."""
    selected: dict[str, dict[str, list[int]]] = {}
    excluded: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dimension in dimensions:
        available: dict[str, list[int]] = {}
        dimension_excluded: dict[str, list[dict[str, Any]]] = {}
        for base_suite in BASE_SUITE_ORDER:
            suite_name = _benchmark_suite(base_suite, dimension)
            suite = _construct_suite(benchmark_dict, suite_name)
            eligible_ids, excluded_tasks = _partition_suite_task_ids(suite, suite_name)
            available[base_suite] = eligible_ids
            dimension_excluded[base_suite] = excluded_tasks
        total_available = sum(len(task_ids) for task_ids in available.values())
        target_count = min(tasks_per_dimension, total_available)

        rng = _dimension_rng(seed, dimension)
        for task_ids in available.values():
            rng.shuffle(task_ids)
        positions = {suite: 0 for suite in BASE_SUITE_ORDER}
        chosen = {suite: [] for suite in BASE_SUITE_ORDER}
        chosen_count = 0
        while chosen_count < target_count:
            made_progress = False
            for base_suite in BASE_SUITE_ORDER:
                position = positions[base_suite]
                if position >= len(available[base_suite]):
                    continue
                chosen[base_suite].append(available[base_suite][position])
                positions[base_suite] += 1
                chosen_count += 1
                made_progress = True
                if chosen_count == target_count:
                    break
            if not made_progress:
                raise RuntimeError(f"failed to select tasks for {dimension}")
        selected[dimension] = chosen
        excluded[dimension] = dimension_excluded
    return selected, excluded


def _select_only_task_ids_by_dimension(
    benchmark_dict: dict[str, Any],
    dimensions: list[str],
    only_tasks: list[tuple[str, int]],
) -> tuple[
    dict[str, dict[str, list[int]]],
    dict[str, dict[str, list[dict[str, Any]]]],
]:
    """Select exactly the requested eligible suite/task pairs."""
    selected = {
        dimension: {base_suite: [] for base_suite in BASE_SUITE_ORDER}
        for dimension in dimensions
    }
    excluded: dict[str, dict[str, list[dict[str, Any]]]] = {
        dimension: {} for dimension in dimensions
    }
    inventories: dict[str, tuple[str, str, Any, set[int]]] = {}
    for dimension in dimensions:
        for base_suite in BASE_SUITE_ORDER:
            suite_name = _benchmark_suite(base_suite, dimension)
            suite = _construct_suite(benchmark_dict, suite_name)
            eligible_ids, excluded_tasks = _partition_suite_task_ids(suite, suite_name)
            inventories[suite_name] = (
                dimension,
                base_suite,
                suite,
                set(eligible_ids),
            )
            excluded[dimension][base_suite] = excluded_tasks

    seen: set[tuple[str, int]] = set()
    for suite_name, task_id in only_tasks:
        selector = (suite_name, task_id)
        if selector in seen:
            raise ValueError(
                f"duplicate --only-task selector: {suite_name}:{task_id}"
            )
        seen.add(selector)
        inventory = inventories.get(suite_name)
        if inventory is None:
            requested = ", ".join(dimensions)
            raise ValueError(
                f"--only-task suite {suite_name!r} is not part of the requested "
                f"dimension(s): {requested}"
            )
        dimension, base_suite, suite, eligible_ids = inventory
        num_tasks = int(suite.get_num_tasks())
        if task_id >= num_tasks:
            raise ValueError(
                f"--only-task {suite_name}:{task_id} is out of range; "
                f"valid IDs are 0-{num_tasks - 1}"
            )
        if task_id not in eligible_ids:
            excluded_task = next(
                item
                for item in excluded[dimension][base_suite]
                if item["task_id"] == task_id
            )
            raise ValueError(
                f"--only-task {suite_name}:{task_id} is excluded from evaluation: "
                f"{excluded_task['reason']}"
            )
        selected[dimension][base_suite].append(task_id)
    return selected, excluded


def _read_bddl_instruction(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    depth = 0
    for line in text.splitlines():
        content = line.split(";", 1)[0]
        depth += content.count("(") - content.count(")")
        if depth < 0:
            break
    if depth != 0:
        raise ValueError(f"unbalanced BDDL parentheses: {path}")
    matches = LANGUAGE_PATTERN.findall(text)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one :language field in {path}, found {len(matches)}"
        )
    instruction = " ".join(matches[0].split())
    if not instruction:
        raise ValueError(f"empty :language field in {path}")
    return instruction


def _normalized_instruction(value: str) -> str:
    return " ".join(value.casefold().split())


def _round_robin_contexts(
    contexts_by_suite: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    max_count = max((len(items) for items in contexts_by_suite.values()), default=0)
    for position in range(max_count):
        for base_suite in BASE_SUITE_ORDER:
            items = contexts_by_suite.get(base_suite, [])
            if position < len(items):
                ordered.append(items[position])
    return ordered


def _build_episode_specs(
    benchmark_dict: dict[str, Any],
    data_path: Path,
    selected_task_ids: dict[str, dict[str, list[int]]],
    dimensions: list[str],
    *,
    trials_per_task: int,
    start_trial: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build trial-, dimension-, and suite-balanced LIBERO-Pro episode specs."""
    end_trial = start_trial + trials_per_task
    contexts_by_dimension: dict[str, list[dict[str, Any]]] = {}
    for dimension in dimensions:
        contexts_by_suite: dict[str, list[dict[str, Any]]] = {}
        for base_suite in BASE_SUITE_ORDER:
            suite_name = _benchmark_suite(base_suite, dimension)
            suite = _construct_suite(benchmark_dict, suite_name)
            suite_contexts: list[dict[str, Any]] = []
            for task_id in selected_task_ids[dimension][base_suite]:
                task = suite.get_task(task_id)
                bddl_path = (
                    data_path / "bddl_files" / suite_name / task.bddl_file
                ).resolve()
                init_path = (
                    data_path / "init_files" / suite_name / task.init_states_file
                ).resolve()
                if not bddl_path.is_file():
                    raise FileNotFoundError(f"missing BDDL file: {bddl_path}")
                if not init_path.is_file():
                    raise FileNotFoundError(f"missing init-state file: {init_path}")
                instruction = _read_bddl_instruction(bddl_path)
                filename_instruction = str(task.language)
                if dimension in {"Semantic", "Task"} and _normalized_instruction(
                    instruction
                ) == _normalized_instruction(filename_instruction):
                    raise ValueError(
                        f"{dimension} instruction did not change in {bddl_path}; "
                        "refusing to evaluate a filename-language no-op"
                    )
                initial_states = np.asarray(suite.get_task_init_states(task_id))
                if initial_states.ndim < 2:
                    raise ValueError(
                        f"invalid init-state array for {suite_name} task {task_id}: "
                        f"shape={initial_states.shape}"
                    )
                if end_trial > len(initial_states):
                    raise ValueError(
                        f"{suite_name} task {task_id} has {len(initial_states)} states; "
                        f"requested trials [{start_trial}, {end_trial})"
                    )
                requested_states = initial_states[start_trial:end_trial]
                if not np.isfinite(requested_states).all():
                    raise ValueError(
                        f"non-finite init state in {suite_name} task {task_id}"
                    )
                suite_contexts.append(
                    {
                        "benchmark": "LIBERO-Pro",
                        "suite": suite_name,
                        "benchmark_suite": suite_name,
                        "base_suite": base_suite,
                        "dimension": dimension,
                        "dimension_slug": _dimension_slug(dimension),
                        "task_id": task_id,
                        "task_name": str(task.name),
                        "filename_instruction": filename_instruction,
                        "instruction": instruction,
                        "bddl_file": str(bddl_path),
                        "initial_states": initial_states,
                    }
                )
            contexts_by_suite[base_suite] = suite_contexts
        contexts_by_dimension[dimension] = _round_robin_contexts(contexts_by_suite)

    specs: list[dict[str, Any]] = []
    max_tasks_in_dimension = max(
        (len(contexts) for contexts in contexts_by_dimension.values()),
        default=0,
    )
    for trial in range(start_trial, end_trial):
        for position in range(max_tasks_in_dimension):
            for dimension in dimensions:
                dimension_contexts = contexts_by_dimension[dimension]
                if position >= len(dimension_contexts):
                    continue
                context = dimension_contexts[position]
                spec = {key: value for key, value in context.items() if key != "initial_states"}
                spec.update(
                    {
                        "key": (
                            f"{context['dimension_slug']}:{context['suite']}:"
                            f"task-{context['task_id']:02d}:trial-{trial:02d}:seed-{seed}"
                        ),
                        "trial": trial,
                        "seed": seed,
                        "initial_state": context["initial_states"][trial]
                        .astype(float)
                        .tolist(),
                    }
                )
                specs.append(spec)
    return specs


def _git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_evaluation_plan(
    output_dir: Path,
    *,
    dimensions: list[str],
    tasks_per_dimension: int | None,
    only_tasks: list[tuple[str, int]],
    trials_per_task: int,
    start_trial: int,
    seed: int,
    selected_task_ids: dict[str, dict[str, list[int]]],
    excluded_tasks: dict[str, dict[str, list[dict[str, Any]]]],
    specs: list[dict[str, Any]],
    source_path: Path,
    data_path: Path,
    environment_generation: dict[str, Any] | None,
    environment_cache: Path | None,
    resume: bool,
) -> None:
    stable_environment_generation = None
    if environment_generation is not None:
        stable_environment_generation = {
            "input_digest": environment_generation["input_digest"],
            "artifact_digest": environment_generation["artifact_digest"],
            "num_init_states": environment_generation["num_init_states"],
            "target_environment": environment_generation["target_environment"],
        }
    stable_plan = {
        "dimensions": dimensions,
        "selection_mode": "exact" if only_tasks else "sampled",
        "task_limit_per_dimension": tasks_per_dimension,
        "only_tasks": [
            f"{suite_name}:{task_id}" for suite_name, task_id in only_tasks
        ],
        "selected_tasks_per_dimension": {
            dimension: sum(
                len(task_ids)
                for task_ids in selected_task_ids[dimension].values()
            )
            for dimension in dimensions
        },
        "excluded_tasks": excluded_tasks,
        "trials_per_task": trials_per_task,
        "total_episodes": len(specs),
        "start_trial": start_trial,
        "seed": seed,
        "source_revision": _git_revision(source_path),
        "data_revision": _git_revision(data_path),
        "environment_generation": stable_environment_generation,
        "episode_keys": [spec["key"] for spec in specs],
    }
    plan_path = output_dir / "evaluation_plan.json"
    if resume and plan_path.exists():
        existing = base._load_json(plan_path)
        previous = {key: existing.get(key) for key in stable_plan}
        if previous != stable_plan:
            raise ValueError(
                "resume arguments or LIBERO-Pro revisions do not match "
                "evaluation_plan.json"
            )
    base._atomic_write_json(
        plan_path,
        {
            "schema_version": "Emerge.libero_pro_evaluation_plan.v2",
            "updated_at": base._utc_now(),
            "environment_cache": (
                str(environment_cache.resolve()) if environment_cache is not None else None
            ),
            **stable_plan,
        },
    )
    for dimension in dimensions:
        suites = {
            base_suite: {
                "benchmark_suite": _benchmark_suite(base_suite, dimension),
                "task_ids": selected_task_ids[dimension][base_suite],
                "excluded_tasks": excluded_tasks[dimension][base_suite],
            }
            for base_suite in BASE_SUITE_ORDER
        }
        base._atomic_write_json(
            output_dir / _dimension_slug(dimension) / "selection.json",
            {
                "schema_version": "Emerge.libero_pro_dimension_selection.v2",
                "dimension": dimension,
                "dimension_slug": _dimension_slug(dimension),
                "selection_mode": "exact" if only_tasks else "sampled",
                "tasks": sum(len(item["task_ids"]) for item in suites.values()),
                "trials_per_task": trials_per_task,
                "seed": seed,
                "suites": suites,
            },
        )


def _run_pro_episode(
    spec: dict[str, Any],
    *,
    args: argparse.Namespace,
    base_driver_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Run through the shared lifecycle while keeping all adaptation local."""
    run_args = argparse.Namespace(**vars(args))
    if run_args.max_steps is None:
        run_args.max_steps = base.MAX_STEPS[spec["base_suite"]]
    result = base._run_episode(
        spec,
        args=run_args,
        base_driver_config=base_driver_config,
        output_dir=output_dir,
        board=None,
    )
    result.update(
        {
            "benchmark": "LIBERO-Pro",
            "base_suite": spec["base_suite"],
            "benchmark_suite": spec["benchmark_suite"],
            "task_name": spec["task_name"],
            "filename_instruction": spec["filename_instruction"],
        }
    )
    base._atomic_write_json(Path(result["attempt_dir"]) / "result.json", result)
    return result


def _iter_episode_results(
    indexed_specs: Iterable[tuple[int, dict[str, Any]]],
    *,
    total_specs: int,
    args: argparse.Namespace,
    base_driver_config: dict[str, Any],
    output_dir: Path,
) -> Iterable[tuple[int, dict[str, Any], dict[str, Any]]]:
    spec_iterator = iter(indexed_specs)

    def announce(index: int, spec: dict[str, Any]) -> None:
        print(
            f"[{index}/{total_specs}] run {spec['key']}: {spec['instruction']}",
            flush=True,
        )

    def run_one(
        index: int, spec: dict[str, Any]
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        return (
            index,
            spec,
            _run_pro_episode(
                spec,
                args=args,
                base_driver_config=base_driver_config,
                output_dir=output_dir,
            ),
        )

    if args.workers == 1:
        for index, spec in spec_iterator:
            announce(index, spec)
            item = run_one(index, spec)
            yield item
            if base._is_infrastructure_error(item[2]) and not args.continue_on_error:
                return
        return

    active: dict[
        futures.Future[tuple[int, dict[str, Any], dict[str, Any]]], None
    ] = {}
    stop_scheduling = False
    with futures.ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix="libero-pro-episode",
    ) as executor:

        def submit_next() -> bool:
            try:
                index, spec = next(spec_iterator)
            except StopIteration:
                return False
            announce(index, spec)
            active[executor.submit(run_one, index, spec)] = None
            return True

        while len(active) < args.workers and submit_next():
            pass
        while active:
            done, _ = futures.wait(active, return_when=futures.FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                item = future.result()
                if base._is_infrastructure_error(item[2]) and not args.continue_on_error:
                    stop_scheduling = True
                yield item
            if not stop_scheduling:
                while len(active) < args.workers and submit_next():
                    pass


def _write_summaries(
    output_dir: Path,
    results: dict[str, dict[str, Any]],
    dimensions: list[str],
) -> None:
    cells: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"episodes": 0, "successes": 0})
    )
    infrastructure_errors = 0
    for result in results.values():
        if base._is_infrastructure_error(result):
            infrastructure_errors += 1
            continue
        dimension = str(result.get("dimension", ""))
        base_suite = str(result.get("base_suite", ""))
        if dimension not in dimensions or base_suite not in BASE_SUITE_ORDER:
            continue
        cell = cells[base_suite][dimension]
        cell["episodes"] += 1
        cell["successes"] += int(bool(result.get("success")))

    dimension_summary: dict[str, dict[str, Any]] = {}
    for dimension in dimensions:
        suites: dict[str, dict[str, Any]] = {}
        for base_suite in BASE_SUITE_ORDER:
            raw = cells[base_suite][dimension]
            episodes = raw["episodes"]
            suites[base_suite] = {
                **raw,
                "success_rate": raw["successes"] / episodes if episodes else None,
            }
        episodes = sum(item["episodes"] for item in suites.values())
        successes = sum(item["successes"] for item in suites.values())
        dimension_summary[dimension] = {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes if episodes else None,
            "suites": suites,
        }
        base._atomic_write_json(
            output_dir / _dimension_slug(dimension) / "summary.json",
            {
                "schema_version": "Emerge.libero_pro_dimension_summary.v1",
                "updated_at": base._utc_now(),
                "dimension": dimension,
                **dimension_summary[dimension],
            },
        )

    total_episodes = sum(item["episodes"] for item in dimension_summary.values())
    total_successes = sum(item["successes"] for item in dimension_summary.values())
    summary = {
        "schema_version": "Emerge.libero_pro_evaluation_summary.v1",
        "updated_at": base._utc_now(),
        "benchmark": "LIBERO-Pro",
        "protocol": "Emerge",
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": total_successes / total_episodes if total_episodes else None,
        "infrastructure_errors": infrastructure_errors,
        "dimensions": dimension_summary,
    }
    base._atomic_write_json(output_dir / "summary.json", summary)

    grid_suites: dict[str, dict[str, Any]] = {}
    for base_suite in BASE_SUITE_ORDER:
        dimension_cells: dict[str, dict[str, Any]] = {}
        for dimension in dimensions:
            raw = cells[base_suite][dimension]
            episodes = raw["episodes"]
            dimension_cells[dimension] = {
                **raw,
                "success_rate": raw["successes"] / episodes if episodes else None,
            }
        grid_suites[base_suite] = dimension_cells
    base._atomic_write_json(
        output_dir / "libero_pro_grid.json",
        {
            "schema_version": "Emerge.libero_pro_grid.v1",
            "updated_at": summary["updated_at"],
            "benchmark": "LIBERO-Pro",
            "protocol": "Emerge",
            "dimensions": dimensions,
            "suites": grid_suites,
            "episodes": total_episodes,
            "successes": total_successes,
            "success_rate": summary["success_rate"],
            "infrastructure_errors": infrastructure_errors,
        },
    )

    header = "| Suite | " + " | ".join(dimensions) + " | Total |"
    separator = "| --- | " + " | ".join("---:" for _ in dimensions) + " | ---: |"
    rows = [header, separator]
    for base_suite in BASE_SUITE_ORDER:
        values: list[str] = []
        suite_episodes = 0
        suite_successes = 0
        for dimension in dimensions:
            cell = grid_suites[base_suite][dimension]
            suite_episodes += cell["episodes"]
            suite_successes += cell["successes"]
            rate = cell["success_rate"]
            values.append("--" if rate is None else f"{rate * 100:.1f}")
        total = (
            "--"
            if suite_episodes == 0
            else f"{suite_successes / suite_episodes * 100:.1f}"
        )
        rows.append(f"| {base_suite} | " + " | ".join(values) + f" | {total} |")
    total_values = []
    for dimension in dimensions:
        rate = dimension_summary[dimension]["success_rate"]
        total_values.append("--" if rate is None else f"{rate * 100:.1f}")
    overall = "--" if summary["success_rate"] is None else f"{summary['success_rate'] * 100:.1f}"
    rows.append("| Total | " + " | ".join(total_values) + f" | {overall} |")
    (output_dir / "libero_pro_grid.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = base._build_parser()
    parser.description = (
        "Evaluate one or more LIBERO-Pro dimensions with Emerge."
    )
    parser.set_defaults(
        driver_config=REPO_ROOT / "dev/libero_pro_eval.json",
        task_ids=None,
        watchdog_ready_timeout_s=600.0,
    )
    parser.add_argument(
        "--dimension",
        action="append",
        required=True,
        type=_parse_dimension,
        help=(
            "Dimension to run: object, position, semantic, task, environment, "
            "available, or all. Repeat to select multiple dimensions."
        ),
    )
    parser.add_argument(
        "--tasks-per-dimension",
        "--count",
        dest="tasks_per_dimension",
        default=None,
        type=int,
        help=(
            "Maximum number of eligible unique tasks selected per dimension (1-40). "
            "Dimensions with audited exclusions are capped at their eligible count."
        ),
    )
    parser.add_argument(
        "--only-task",
        action="append",
        default=[],
        type=_parse_only_task,
        metavar="DERIVED_SUITE:TASK_ID",
        help=(
            "Run exactly one derived-suite task. Repeat to select multiple tasks; "
            "cannot be combined with --count."
        ),
    )
    for action in parser._actions:
        if action.dest in {"suite", "task_ids", "full"}:
            action.help = argparse.SUPPRESS
        elif action.dest == "policy_backend":
            action.choices = ("vla",)
            action.default = "vla"
            action.help = "LIBERO-Pro currently supports the VLA backend only."
        elif action.dest in {
            "wam_server_url",
            "wam_conditioning_mode",
            "skip_wam_server_check",
        }:
            action.help = argparse.SUPPRESS
    return parser


def main() -> int:
    _install_robosuite_compat()
    parser = _build_parser()
    args = parser.parse_args()
    if args.suite or args.task_ids is not None or args.full:
        parser.error(
            "LIBERO-Pro selection is dimension-only; remove --suite, --task-ids, "
            "and --full"
        )
    if args.only_task and args.tasks_per_dimension is not None:
        parser.error("--only-task cannot be combined with --tasks-per-dimension/--count")
    if not args.only_task and args.tasks_per_dimension is None:
        parser.error("one of --only-task or --tasks-per-dimension/--count is required")
    if args.tasks_per_dimension is not None:
        if args.tasks_per_dimension <= 0:
            parser.error("--tasks-per-dimension/--count must be positive")
        if args.tasks_per_dimension > MAX_TASKS_PER_DIMENSION:
            parser.error(
                f"--tasks-per-dimension/--count cannot exceed {MAX_TASKS_PER_DIMENSION}"
            )
    if args.trials_per_task <= 0:
        parser.error("--trials-per-task must be positive")
    if args.start_trial < 0:
        parser.error("--start-trial cannot be negative")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.num_steps_wait < 0:
        parser.error("--num-steps-wait cannot be negative")
    if args.max_agent_feedback_rounds < 0:
        parser.error("--max-agent-feedback-rounds cannot be negative")
    if args.stream:
        parser.error("--stream is not supported by the initial LIBERO-Pro entrypoint")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        base._resolve_path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "artifacts/libero_pro_agent_eval" / timestamp
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
        libero_config.get("libero_source_path", "third_party/libero_pro")
    )
    data_path = base._resolve_path(
        libero_config.get("libero_data_path", "third_party/libero_pro_data")
    )
    environment_cache_root = base._resolve_path(
        libero_config.get(
            "libero_environment_cache_path",
            "artifacts/libero_pro_eval_generated",
        )
    )

    try:
        effective_data_path = data_path
        environment_cache = None
        environment_generation = None
        if _requests_environment(args.dimension) and _dimension_missing_paths(
            data_path, "Environment"
        ):
            environment_cache, environment_generation = prepare_environment_cache(
                source_path,
                environment_cache_root,
                protected_roots=(data_path,),
            )
            effective_data_path = prepare_data_overlay(
                output_dir,
                data_path,
                environment_cache,
            )
        _prepare_libero_pro_config(output_dir, source_path, effective_data_path)
        dimensions = _resolve_dimensions(args.dimension, effective_data_path)
        benchmark_dict = _load_benchmark_dict(source_path)
        if args.only_task:
            selected_task_ids, excluded_tasks = _select_only_task_ids_by_dimension(
                benchmark_dict,
                dimensions,
                args.only_task,
            )
        else:
            selected_task_ids, excluded_tasks = _select_task_ids_by_dimension(
                benchmark_dict,
                dimensions,
                tasks_per_dimension=args.tasks_per_dimension,
                seed=args.seed,
            )
        specs = _build_episode_specs(
            benchmark_dict,
            effective_data_path,
            selected_task_ids,
            dimensions,
            trials_per_task=args.trials_per_task,
            start_trial=args.start_trial,
            seed=args.seed,
        )
        _write_evaluation_plan(
            output_dir,
            dimensions=dimensions,
            tasks_per_dimension=args.tasks_per_dimension,
            only_tasks=args.only_task,
            trials_per_task=args.trials_per_task,
            start_trial=args.start_trial,
            seed=args.seed,
            selected_task_ids=selected_task_ids,
            excluded_tasks=excluded_tasks,
            specs=specs,
            source_path=source_path,
            data_path=data_path,
            environment_generation=environment_generation,
            environment_cache=environment_cache,
            resume=args.resume,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    selection_detail = (
        "only_tasks="
        + ",".join(
            f"{suite_name}:{task_id}" for suite_name, task_id in args.only_task
        )
        if args.only_task
        else f"task_limit_per_dimension={args.tasks_per_dimension}"
    )
    print(
        f"Prepared {len(specs)} LIBERO-Pro episode(s): dimensions={dimensions}, "
        f"{selection_detail}, trials_per_task={args.trials_per_task}, "
        f"workers={args.workers}, schedule=trial-dimension-suite-round-robin"
    )
    for dimension in dimensions:
        suite_counts = ", ".join(
            f"{base_suite}={len(selected_task_ids[dimension][base_suite])}"
            for base_suite in BASE_SUITE_ORDER
        )
        excluded_count = sum(
            len(tasks) for tasks in excluded_tasks[dimension].values()
        )
        print(f"  {dimension}: {suite_counts}; excluded={excluded_count}")
    print(f"Protocol: Emerge + pi0.5 | Output: {output_dir}")
    if args.dry_run:
        displayed_specs = specs if len(specs) <= 200 else specs[:20]
        for spec in displayed_specs:
            print(
                f"{spec['key']} | [{spec['base_suite']}] | "
                f"{spec['instruction']} | {spec['bddl_file']}"
            )
        omitted = len(specs) - len(displayed_specs)
        if omitted:
            print(
                f"... {omitted} additional episode(s) recorded in "
                f"{output_dir / 'evaluation_plan.json'}"
            )
        return 0

    if (
        not args.skip_policy_server_check
        and not base._server_is_ready(args.vla_server_url or args.policy_server_url)
    ):
        server_url = args.vla_server_url or args.policy_server_url
        parser.error(
            f"policy server is not reachable at {server_url}; "
            "start external_model_server/openpi_batch_server.py first or pass "
            "--skip-policy-server-check"
        )

    stored_results = base._read_results(results_path)
    completed: dict[str, dict[str, Any]] = {}
    pending_specs: list[tuple[int, dict[str, Any]]] = []
    for index, spec in enumerate(specs, start=1):
        previous = stored_results.get(spec["key"]) if args.resume else None
        if previous is not None and not base._is_infrastructure_error(previous):
            completed[spec["key"]] = previous
            print(f"[{index}/{len(specs)}] skip completed {spec['key']}")
            continue
        if previous is not None:
            print(f"[{index}/{len(specs)}] retry infrastructure error {spec['key']}")
        pending_specs.append((index, spec))

    infrastructure_failed = False
    for _, spec, result in _iter_episode_results(
        pending_specs,
        total_specs=len(specs),
        args=args,
        base_driver_config=base_driver_config,
        output_dir=output_dir,
    ):
        with results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        completed[spec["key"]] = result
        _write_summaries(output_dir, completed, dimensions)
        print(
            f"  success={result['success']} reason={result['termination_reason']} "
            f"steps={result['action_steps']} duration={result['duration_s']:.1f}s"
        )
        infrastructure_failed = (
            infrastructure_failed or base._is_infrastructure_error(result)
        )

    _write_summaries(output_dir, completed, dimensions)
    if infrastructure_failed and not args.continue_on_error:
        print(
            "Stopped scheduling new episodes after infrastructure error.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
