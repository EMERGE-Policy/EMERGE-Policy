"""Read-only progress and rough ETA for an existing LIBERO evaluation directory."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import time


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def duration(seconds: float) -> str:
    minutes, seconds = divmod(int(max(0, seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def report(root: Path, workers: int) -> None:
    plan_path = root / "evaluation_plan.json"
    plan = load_json(plan_path)
    if not isinstance(plan.get("episode_keys"), list):
        raise ValueError(f"Missing/invalid evaluation plan: {plan_path}")
    expected = set(plan["episode_keys"])
    results = {}
    result_path = root / "results.jsonl"
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue  # The evaluator may be writing its last line.
            if isinstance(item, dict) and item.get("episode_key") in expected:
                results[item["episode_key"]] = item
    total, completed = len(expected), len(results)
    successes = sum(bool(item.get("success")) for item in results.values())
    elapsed = time.time() - plan_path.stat().st_mtime
    print(f"\n{datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"Completed: {completed}/{total} | Successes: {successes} | "
          f"Elapsed since plan preparation/resume: {duration(elapsed)}")
    samples = [item.get("duration_s") for item in results.values()]
    samples = [float(x) for x in samples if isinstance(x, (int, float))
               and math.isfinite(x) and x > 0]
    if completed == total:
        print("Remaining: 00:00:00")
    elif samples:
        mean = statistics.mean(samples)
        estimate = (total - completed) * mean / workers
        print(f"Mean episode: {duration(mean)} | Rough remaining: {duration(estimate)} "
              f"({workers} workers; queue/startup/LLM variation affects this estimate)")
    else:
        print("Remaining: unknown (no completed episodes yet)")

    # Select the latest attempt for each suite/task/trial, ignoring older retries.
    active = {}
    for path in sorted(root.glob("*/task_*/trial_*/attempt_*/status.json")):
        status = load_json(path)
        meta = status.get("metadata") or {}
        trial_parts = path.parents[1].name.split("_")
        if len(trial_parts) != 4 or "task_id" not in meta:
            continue
        key = (f"{meta.get('suite')}:task-{meta['task_id']:02d}:"
               f"trial-{int(trial_parts[1]):02d}:seed-{trial_parts[3]}")
        if key in expected and key not in results:
            active[key] = status
    for key, status in active.items():
        print(f"  {key}: actions={status.get('action_steps', 0)}/"
              f"{status.get('max_action_steps', '?')} ready={status.get('ready')} "
              f"success={status.get('success')} done={status.get('done')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, required=True, help="Use the evaluator's actual worker count")
    parser.add_argument("--watch", type=float, default=0, help="Refresh interval in seconds (0: print once)")
    args = parser.parse_args()
    if args.workers < 1 or not math.isfinite(args.watch) or args.watch < 0:
        parser.error("workers must be positive and watch must be finite/non-negative")
    try:
        while True:
            report(args.output_dir.expanduser().resolve(), args.workers)
            if args.watch == 0:
                break
            time.sleep(max(1, args.watch))
    except KeyboardInterrupt:
        pass  # Stopping this monitor does not signal or stop the evaluator.
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
