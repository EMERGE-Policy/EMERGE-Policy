"""Real LIBERO/WAM smoke test without an outer LLM; requires real model servers.

Run from the repository: python -m dev.smoke_robometer_wam --help.
This executes a bounded action chunk, not a task-success benchmark.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path

import numpy as np

from scripts.Libero_eval.eval_libero_agent import (
    DEFAULT_LIBERO_SOURCE,
    REPO_ROOT,
    _build_episode_specs,
    _prepare_libero_config,
)
from robot.drivers.mujoco_driver import LiberoMujocoDriver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libero-source", type=Path, default=DEFAULT_LIBERO_SOURCE)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--cosmos-port", type=int, default=8003)
    parser.add_argument("--score-mode", choices=("robometer", "joint_value"), default="robometer")
    parser.add_argument("--frames-only", action="store_true", help="Save real RGB observations without querying WAM")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.steps < 1:
        parser.error("--steps must be positive")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    _prepare_libero_config(output, args.libero_source.resolve())
    spec = _build_episode_specs(suite_names=["libero_object"], task_ids_text=str(args.task_id),
                                trials_per_task=1, start_trial=0, seed=1)[0]
    config = json.loads((REPO_ROOT / "dev/libero_robometer_eval.json").read_text())
    config["libero"].update(bddl_root=spec["bddl_root"], bddl_file_name=spec["bddl_file"],
                             libero_source_path=str(args.libero_source.resolve()))
    # Only the two policy cameras are needed for this transport/execution check.
    config["cameras"] = {key: value for key, value in config["cameras"].items()
                         if key in {"agentview", "robot0_eye_in_hand"}}
    config["wam"]["search"].update(num_candidates=args.candidates, score_mode=args.score_mode)
    config["wam"]["discovery"] = {"start_port": args.cosmos_port, "end_port": args.cosmos_port}
    config["evaluation"] = {"policy_backend": "wam", "initial_state": spec["initial_state"],
                             "num_steps_wait": 10, "seed": 1, "max_action_steps": args.steps + 4}
    driver = LiberoMujocoDriver(workspace=output, **config)
    summary = {"kind": "real_libero_frames" if args.frames_only else "real_wam_bounded_smoke",
               "task": spec["instruction"], "score_mode": args.score_mode,
               "requested_steps": args.steps, "num_candidates": args.candidates}
    try:
        driver.load_environment()
        env = driver._environment
        initial = env.latest_obs["agentview_image"].copy()
        if args.frames_only:
            for _ in range(3):
                env.step([0.0] * 6 + [-1.0], publish_observation=False)
            # Raw LIBERO observations use the opposite vertical orientation to
            # the training images. These are actual time steps from ONE camera.
            pair = np.stack([np.flipud(initial), np.flipud(env.latest_obs["agentview_image"])])
            np.save(output / "libero_two_frames.npy", pair, allow_pickle=False)
            summary.update(shape=list(pair.shape), dtype=str(pair.dtype))
        else:
            result = driver._wam.execute(spec["instruction"], args.steps,
                                          task_instruction=spec["instruction"])
            summary["result"] = dataclasses.asdict(result)
            if not result.success or (result.total_steps != args.steps and not result.task_success):
                raise RuntimeError(f"WAM smoke failed: {result}")
            for decision in result.search_decisions:
                scores = decision["candidate_scores"]
                if decision["selected_index"] != int(np.argmax(scores)):
                    raise AssertionError("Planner did not select the highest candidate score")
                if args.score_mode == "robometer" and decision["scoring"]["backend"] != "robometer":
                    raise AssertionError("Wrong scoring backend")
            summary["completed"] = True
        print(json.dumps(summary, indent=2))
    finally:
        driver.close()
        (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
