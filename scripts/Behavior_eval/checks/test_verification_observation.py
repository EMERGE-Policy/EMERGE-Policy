"""Checks for the dedicated four-view task-verification manifest."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from Emerge.agent.visual_monitor import VisualMonitor
from Emerge.subagents import ImageContent, TextContent
from Emerge.subagents.task_verification.tools.observation import ObservationStore


class VerificationObservationStoreTest(unittest.TestCase):
    @staticmethod
    def _write_manifest(workspace: Path, relative_path: str, names: list[str], schema: str) -> None:
        views = []
        for name in names:
            image_path = workspace / "artifacts" / "test_cameras" / f"{name}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(20, 40, 60)).save(image_path)
            views.append(
                {
                    "name": name,
                    "image_path": image_path.relative_to(workspace).as_posix(),
                }
            )
        manifest_path = workspace / relative_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": schema,
                    "reference_view": names[0],
                    "views": views,
                }
            ),
            encoding="utf-8",
        )

    def test_behavior_verification_manifest_supplies_exactly_four_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            self._write_manifest(
                workspace,
                "artifacts/observations/observation.json",
                ["head", "left_wrist", "right_wrist"],
                "legacy",
            )
            expected = ["room_overview", "table_overhead", "table_side", "head"]
            VerificationObservationStoreTest._write_manifest(
                workspace,
                "artifacts/verification_observations/observation.json",
                expected,
                "Emerge.behavior_verification_observation.v1",
            )

            observation = ObservationStore(workspace).load()

            self.assertEqual([view.name for view in observation.views], expected)
            self.assertEqual(observation.reference_view, "room_overview")

    def test_existing_benchmarks_fall_back_to_original_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            self._write_manifest(
                workspace,
                "artifacts/observations/observation.json",
                ["front"],
                "legacy",
            )

            observation = ObservationStore(workspace).load()

            self.assertEqual([view.name for view in observation.views], ["front"])


class _ImmediateFuture:
    def result(self):
        return {"outcome": "not_achieved"}


class VisualMonitorCameraSelectionTest(unittest.TestCase):
    def _captured_content(
        self,
        workspace: Path,
        *,
        with_verification_manifest: bool,
    ):
        (workspace / "PLAN.md").write_text("plan", encoding="utf-8")
        (workspace / "ACTION.md").write_text("action", encoding="utf-8")
        VerificationObservationStoreTest._write_manifest(
            workspace,
            "artifacts/observations/observation.json",
            ["head", "left_wrist", "right_wrist"],
            "legacy",
        )
        if with_verification_manifest:
            VerificationObservationStoreTest._write_manifest(
                workspace,
                "artifacts/verification_observations/observation.json",
                ["room_overview", "table_overhead", "table_side", "head"],
                "Emerge.behavior_verification_observation.v1",
            )

        monitor = VisualMonitor(
            workspace=workspace,
            provider=object(),
            model="test",
            poll_interval_seconds=1,
            verification_timeout_seconds=1,
            confirmations=1,
        )
        monitor._main_loop = object()
        subgoal = SimpleNamespace(
            plan_revision="plan-1",
            step_id="step-1",
            subgoal="approach the radio",
            done_criterion="radio is within bimanual reach",
        )
        with (
            patch(
                "Emerge.agent.visual_monitor.read_current_subgoal",
                return_value=subgoal,
            ),
            patch.object(
                monitor,
                "_read_pending_action",
                return_value={"id": "action-1"},
            ),
            patch(
                "Emerge.agent.visual_monitor.concurrent.futures.Future",
                _ImmediateFuture,
            ),
        ):
            monitor._poll_once()
        task, _future = monitor._verification_requests.get_nowait()
        return task.content

    def test_behavior_visual_monitor_receives_all_four_named_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            content = self._captured_content(
                Path(temporary),
                with_verification_manifest=True,
            )
        labels = [
            item.text.removeprefix("Camera view: ")
            for item in content
            if isinstance(item, TextContent) and item.text.startswith("Camera view: ")
        ]
        images = [item for item in content if isinstance(item, ImageContent)]
        self.assertEqual(
            labels,
            ["room_overview", "table_overhead", "table_side", "head"],
        )
        self.assertEqual(len(images), 4)

    def test_legacy_visual_monitor_keeps_reference_view_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            content = self._captured_content(
                Path(temporary),
                with_verification_manifest=False,
            )
        labels = [
            item.text.removeprefix("Camera view: ")
            for item in content
            if isinstance(item, TextContent) and item.text.startswith("Camera view: ")
        ]
        images = [item for item in content if isinstance(item, ImageContent)]
        self.assertEqual(labels, ["head"])
        self.assertEqual(len(images), 1)


if __name__ == "__main__":
    unittest.main()
