"""Read the current subgoal from PLAN.md."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from Emerge.agent.tools.update_plan import UpdatePlanTool


@dataclass(frozen=True, slots=True)
class CurrentSubgoal:
    plan_revision: str
    step_id: str
    subgoal: str
    done_criterion: str


def read_current_subgoal(plan_file: Path) -> CurrentSubgoal | None:
    content = plan_file.read_text(encoding="utf-8")
    plan = UpdatePlanTool._parse(content)
    revision = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    if plan["branch_stack"]:
        item = plan["branch_stack"][-1]
        step_id = f"branch:{item['id']}"
    else:
        pointer = plan["pointer"]
        if pointer > len(plan["main_line"]):
            return None
        item = plan["main_line"][pointer - 1]
        step_id = f"main:{item['id']}"

    return CurrentSubgoal(
        plan_revision=revision,
        step_id=step_id,
        subgoal=item["subgoal"],
        done_criterion=item["done_criterion"],
    )
