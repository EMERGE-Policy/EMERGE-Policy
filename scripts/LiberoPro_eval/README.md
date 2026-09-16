# Emerge-Policy LIBERO-Pro Evaluation Guide

Select tasks with `--dimension` and `--count` or `--only-task`.

## Prerequisites

Run from the repository root:

```bash
cd /path/to/EMERGE-Policy
conda activate EmergePolicy
```

Required checkouts:

```text
third_party/libero_pro       # https://github.com/Zxy-MLlab/LIBERO-PRO
third_party/libero_pro_data  # https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro
```

Do not install LIBERO-Pro's pinned `requirements.txt` into the EmergePolicy
environment. The evaluator uses the existing robosuite compatibility layer and
loads LIBERO-Pro from `third_party/libero_pro`.

## Start services (terminal 1)

### VLA

```bash
OPENPI_GPU=0 \
VGGT_GPU=1 \
SAM3_GPU=2 \
bash scripts/model_server/start_external_model_servers.sh --services openpi,vggt,sam3
```

Batch size defaults to 1. For multiple workers, see
[batch settings](../model_server/README.md#optional-batching-for-multiple-workers).
Health checks are in the [service guide](../model_server/README.md#2-check-services).

### WAM

LIBERO-Pro currently supports VLA only. The WAM backend is not available for
this evaluation entrypoint.

## Quick evaluation (terminal 2)

```bash
conda activate EmergePolicy
python scripts/LiberoPro_eval/eval_libero_pro_agent.py \
  --policy-backend vla \
  --dimension object \
  --count 1 \
  --trials-per-task 1 \
  --workers 1 \
  --record-video \
  --output-dir artifacts/libero_pro_agent_eval/vla_smoke
```

Add `--max-steps 16` for a short execution check or `--dry-run` to list tasks.

## Dimensions

| CLI name | LIBERO-Pro suffix |
| --- | --- |
| `object` | `_object` |
| `position` | `_swap` |
| `semantic` | `_lan` |
| `task` | `_task` |
| `environment` | `_env` |

`--dimension` is required and can be repeated. `available` and `all` both cover
all five official dimensions. The current official dataset checkout omits the
four Environment suite directories, so the evaluator generates them on first
use with the official `perturbation.py`, `ood_environment.yaml`, and
`generate_init_states.py` behavior.

Generated Environment data is cached at
`artifacts/libero_pro_eval_generated` (configurable with
`libero.libero_environment_cache_path`). Every task receives the official 50
init states. A run-local symlink overlay combines that cache with the frozen
four-dimensional dataset; neither `third_party/libero_pro` nor
`third_party/libero_pro_data` is modified. First-time generation creates 2,000
MuJoCo environments and can take a while; later runs validate and reuse it.

## Audited task exclusions

The upstream suites register 200 BDDLs, but this evaluator scores 189. Eleven
tasks are always excluded by derived suite plus BDDL filename. The
current zero-based task IDs are included below for inspection; selection does
not rely on IDs, so a task-order change cannot silently skip a different BDDL.

| Derived suite | Task ID | BDDL | Reason for exclusion |
| --- | ---: | --- | --- |
| `libero_spatial_swap` | 3 | `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl` | User-requested stability exclusion. The elevated bowl-on-cookie-box grasp was unreliable in the original run, although a later targeted retry succeeded in 155 action steps. The BDDL language and goal are consistent, so this is not an upstream semantic defect. |
| `libero_10_object` | 0 | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl` | The Object perturbation requires manipulating `bigger_alphabet_soup`; its oversized body cannot be reliably enclosed by the Panda gripper. |
| `libero_10_object` | 5 | `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl` | The instruction requests a book, but no book is instantiated and the success goal requires `black_bowl_1`. A language-following policy cannot identify the verifier target. |
| `libero_10_object` | 7 | `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl` | The Object perturbation requires manipulating `bigger_alphabet_soup`; its oversized body cannot be reliably enclosed by the Panda gripper. |
| `libero_10_object` | 9 | `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl` | The instruction identifies a yellow-and-white mug, while the success goal requires `red_coffee_mug_1` and another mug is also present. The verifier target is ambiguous from the instruction. |
| `libero_object_object` | 0 | `pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl` | The Object perturbation replaces alphabet soup with `bigger_alphabet_soup`; its oversized body cannot be reliably enclosed by the Panda gripper. |
| `libero_10_env` | 0 | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl` | No Environment perturbation occurs: the source workspace and fixed replacement are both `living_room_table`. |
| `libero_10_env` | 1 | `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.bddl` | Same `living_room_table` Environment no-op. |
| `libero_10_env` | 4 | `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl` | Same `living_room_table` Environment no-op. |
| `libero_10_env` | 6 | `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl` | Same `living_room_table` Environment no-op. |
| `libero_10_env` | 7 | `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl` | Same `living_room_table` Environment no-op. |

Two Object tasks are excluded because their language does not reliably name the
object required by the success predicate. Three more Object tasks are excluded
because their success predicates require the oversized `bigger_alphabet_soup`,
which cannot be reliably enclosed by the Panda gripper. The Position task is a
requested operational stability exclusion, not a broken BDDL. The five
Environment tasks are valid base tasks, but they are excluded from the
Environment score because the generated BDDL is byte-for-byte identical to its
source and therefore does not measure environment generalization. The full
scored inventory is:

| Dimension | Scored tasks |
| --- | ---: |
| Object | 35 |
| Position | 39 |
| Semantic | 40 |
| Task | 40 |
| Environment | 35 |
| **Total** | **189** |

Environment cache generation intentionally remains compatible with the
official 40-task suite registration and still creates all 40 BDDLs and init
state files. The five no-op tasks are filtered only when evaluation tasks are
selected and never enter score denominators. Exact exclusions and reasons are
also recorded in `evaluation_plan.json` and each dimension's `selection.json`.

## Selection semantics

`--count` / `--tasks-per-dimension` is the maximum number of eligible unique
tasks selected for each dimension. Raw dimensions register 40 tasks: 10 tasks
in each of `libero_goal`, `libero_spatial`, `libero_10`, and `libero_object`.
Audited dimensions are capped at their eligible inventory, so `--count 40`
selects 35 Object, 39 Position, 40 Semantic, 40 Task, and 35 Environment tasks.
For exact diagnostic reruns, use one or more `--only-task` selectors instead of
`--count`; the two selection modes are mutually exclusive.

Total episodes are:

```text
sum of selected eligible tasks in requested dimensions x trials per task
```

Task selection is deterministic for a given `--seed` and balanced round-robin
across the four base suites. Scheduling is round-robin by trial, dimension, and
suite so interrupted runs retain broad coverage.

## Dry runs

One selected task per dimension:

```bash
python scripts/LiberoPro_eval/eval_libero_pro_agent.py \
  --policy-backend vla \
  --dimension available \
  --count 1 \
  --trials-per-task 1 \
  --dry-run
```

Validate all 189 scored task variants (after one-time Environment preparation):

```bash
python scripts/LiberoPro_eval/eval_libero_pro_agent.py \
  --policy-backend vla \
  --dimension available \
  --count 40 \
  --trials-per-task 1 \
  --dry-run
```

## Targeted reruns

Use `--only-task DERIVED_SUITE:TASK_ID` to run an exact task, and repeat the
option to build a small multi-task run. The derived suite must belong to a
requested dimension. Duplicate, nonexistent, cross-dimension, and audited-out
tasks are rejected before an evaluation plan is written.

For example, this reruns two Position failures without scheduling any other
tasks:

```bash
python scripts/LiberoPro_eval/eval_libero_pro_agent.py \
  --policy-backend vla \
  --dimension position \
  --only-task libero_10_swap:2 \
  --only-task libero_10_swap:8 \
  --seed 7 \
  --start-trial 0 \
  --trials-per-task 1 \
  --workers 1 \
  --output-dir artifacts/libero_pro_agent_eval/position_targeted_rerun_remaining
```

Add `--only-task libero_goal_swap:9` to the same command if the wine-rack
failure should be included. Add `--dry-run` to inspect the exact episode keys
without executing robot actions. `libero_spatial_swap:3` is now part of the
global exclusion policy and is rejected by targeted selection as well.

## Smoke evaluation

Select one task from every base suite in every dimension:

```bash
python scripts/LiberoPro_eval/eval_libero_pro_agent.py \
  --policy-backend vla \
  --dimension available \
  --count 4 \
  --trials-per-task 1 \
  --workers 4 \
  --output-dir artifacts/libero_pro_agent_eval/smoke
```

Start the VLA service stack before running evaluation commands.

## Larger evaluations

All 189 scored tasks, one trial each:

```bash
python scripts/LiberoPro_eval/eval_libero_pro_agent.py \
  --policy-backend vla \
  --dimension available \
  --count 40 \
  --trials-per-task 1 \
  --workers 4 \
  --output-dir artifacts/libero_pro_agent_eval/pilot189
```

The filtered full scale contains 9,450 episodes (189 tasks x 50 trials):

```bash
python scripts/LiberoPro_eval/eval_libero_pro_agent.py \
  --policy-backend vla \
  --dimension all \
  --count 40 \
  --trials-per-task 50 \
  --workers 4 \
  --output-dir artifacts/libero_pro_agent_eval/full
```

This entrypoint evaluates the Emerge outer-agent protocol with pi0.5 as its
VLA policy. Label its results as `Emerge on LIBERO-Pro`; they are not a
policy-only reproduction of the paper leaderboard.

## Output

Each run writes:

- `evaluation_plan.json`: dimensions, exact episode keys, source/data revisions,
  and resume-critical arguments.
- `<dimension>/selection.json`: selected task IDs grouped by base suite.
- `results.jsonl`: append-only episode results and resume record.
- `summary.json`: success rates grouped by dimension and base suite.
- `libero_pro_grid.json` and `libero_pro_grid.md`: suite-by-dimension result grid.
- `<dimension>/<derived_suite>/task_XX/...`: per-attempt logs, workspace, status,
  driver configuration, and optional video.

Infrastructure errors are reported separately and excluded from success-rate
denominators. `--resume` retries infrastructure errors and skips completed
episodes when its plan and pinned revisions match.

The evaluator reads the task instruction from each BDDL `:language` field. This
is required for Semantic and Task perturbations; the filename-derived LIBERO
language is retained only as result provenance.
