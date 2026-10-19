"""No-simulator checks for the active LightNav/R1Pro handoff boundary."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from Emerge.subagents.object_location.tools.observation import CameraObservation, CameraView
from integrations.lightnav_behavior.arrival import verify_radio_handoff

from integrations.lightnav_behavior.closed_loop import run_closed_loop
from integrations.lightnav_behavior.r1pro_adapter import frame_action_steps, waypoint_action


class R1ProAdapterTests(unittest.TestCase):
    def test_frame_cadence_matches_30_action_hz_and_4_model_hz(self):
        self.assertEqual([frame_action_steps(i) for i in range(4)], [7, 8, 7, 8])
        self.assertEqual(sum(frame_action_steps(i) for i in range(4)), 30)

    def test_waypoint_conversion_preserves_all_non_base_joints(self):
        hold = [0.0, 0.0, 0.0] + [float(i) / 100 for i in range(20)]
        action, metadata = waypoint_action((0.0875, -0.0875, 0.0), hold, action_steps=7)
        self.assertEqual(len(action), 23)
        self.assertEqual(action[3:], hold[3:])
        self.assertAlmostEqual(action[0], 0.5)
        self.assertAlmostEqual(action[1], -0.5)
        self.assertFalse(metadata["clipped"])

    def test_velocity_is_clipped_and_nan_rejected(self):
        action, metadata = waypoint_action((4.0, 0.0, 0.0), [0.0] * 23, action_steps=8)
        self.assertEqual(action[0], 0.6)
        self.assertTrue(metadata["clipped"])
        with self.assertRaises(ValueError):
            waypoint_action((float("nan"), 0.0, 0.0), [0.0] * 23, action_steps=8)


class ClosedLoopTests(unittest.TestCase):
    def test_multiple_calls_append_frames_instructions_and_final_observations(self):
        with TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.jsonl"
            frames = iter([SimpleNamespace(jpeg=str(i).encode(), observation_seq=i, episode_id=1)
                           for i in (1, 8, 9, 16)])
            client = SimpleNamespace(predict=lambda *_: SimpleNamespace(
                first_waypoint=(0.1, 0.0, 0.0), stop=False, visible=None, latency_ms=None))
            for call_id, instruction in enumerate(("go left", "go right"), 1):
                result = run_closed_loop(
                    instruction=instruction, max_steps=7, frame_source=lambda: next(frames),
                    execute_waypoint=lambda _p, n: {"status": "step_completed", "completed_steps": n},
                    steps_per_frame=frame_action_steps, client=client, trace_path=trace,
                    max_wall_time_s=10, append_trace=True, call_id=call_id,
                )
                self.assertTrue(result["post_budget_observation_confirmed"])
            self.assertEqual([p.read_bytes() for p in sorted((Path(temporary) / "lightnav_frames").glob("*.jpg"))],
                             [b"1", b"8", b"9", b"16"])
            import json
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual([row["call_id"] for row in records], [1, 1, 2, 2])
            self.assertEqual([row["instruction"] for row in records], ["go left", "go left", "go right", "go right"])

    def test_stale_post_action_frame_is_not_a_confirmed_segment(self):
        frame = SimpleNamespace(jpeg=b"same", observation_seq=1, episode_id=1)
        client = SimpleNamespace(predict=lambda *_: SimpleNamespace(
            first_waypoint=(0.1, 0.0, 0.0), stop=False, visible=None, latency_ms=None))
        with TemporaryDirectory() as temporary:
            result = run_closed_loop(
                instruction="approach", max_steps=7, frame_source=lambda: frame,
                execute_waypoint=lambda _p, n: {"status": "step_completed", "completed_steps": n},
                steps_per_frame=frame_action_steps, client=client,
                trace_path=Path(temporary) / "trace.jsonl", max_wall_time_s=10,
            )
        self.assertEqual(result["finish_reason"], "observation_error")
        self.assertFalse(result["post_budget_observation_confirmed"])

    def test_keeps_one_session_and_counts_confirmed_gateway_steps(self):
        frames = iter([
            SimpleNamespace(jpeg=b"one", observation_seq=1, episode_id=1),
            SimpleNamespace(jpeg=b"two", observation_seq=9, episode_id=1),
            SimpleNamespace(jpeg=b"three", observation_seq=17, episode_id=1),
        ])

        class Client:
            calls = 0

            def predict(self, jpeg, instruction):
                self.calls += 1
                return SimpleNamespace(first_waypoint=(0.1, 0.0, 0.0),
                                       stop=self.calls == 3, visible=None, latency_ms=2.0)

        client = Client()
        requests = []

        def execute(waypoint, steps):
            requests.append((waypoint, steps))
            return {"completed_steps": steps, "status": "step_completed", "action": {}}

        with TemporaryDirectory() as temporary:
            result = run_closed_loop(
                instruction="go to the radio", max_steps=100, frame_source=lambda: next(frames),
                execute_waypoint=execute, steps_per_frame=frame_action_steps,
                client=client, trace_path=Path(temporary) / "trace.jsonl", max_wall_time_s=10,
            )
            self.assertEqual(len(list((Path(temporary) / "lightnav_frames").glob("*.jpg"))), 3)
        self.assertEqual(result["finish_reason"], "model_stop_candidate")
        self.assertEqual(result["nav_steps"], 15)
        self.assertEqual(client.calls, 3)
        self.assertEqual([steps for _waypoint, steps in requests], [7, 8])

    def test_fails_closed_when_gateway_cannot_finish_action(self):
        frame = SimpleNamespace(jpeg=b"one", observation_seq=1, episode_id=1)
        client = SimpleNamespace(predict=lambda *_: SimpleNamespace(
            first_waypoint=(0.1, 0.0, 0.0), stop=False, visible=None, latency_ms=None))
        with TemporaryDirectory() as temporary:
            result = run_closed_loop(
                instruction="go to the radio", max_steps=100, frame_source=lambda: frame,
                execute_waypoint=lambda *_: {"completed_steps": 2, "status": "interrupted"},
                steps_per_frame=frame_action_steps, client=client,
                trace_path=Path(temporary) / "trace.jsonl", max_wall_time_s=10,
            )
        self.assertEqual(result["finish_reason"], "gateway_interrupted")
        self.assertEqual(result["nav_steps"], 2)

    def test_budget_end_saves_post_action_frame(self):
        frames = iter([
            SimpleNamespace(jpeg=b"before", observation_seq=1, episode_id=1),
            SimpleNamespace(jpeg=b"after", observation_seq=9, episode_id=1),
        ])
        client = SimpleNamespace(predict=lambda *_: SimpleNamespace(
            first_waypoint=(0.1, 0.0, 0.0), stop=False, visible=None, latency_ms=None))
        with TemporaryDirectory() as temporary:
            result = run_closed_loop(
                instruction="go to the radio", max_steps=7, frame_source=lambda: next(frames),
                execute_waypoint=lambda _point, steps: {
                    "completed_steps": steps, "status": "step_completed"},
                steps_per_frame=frame_action_steps, client=client,
                trace_path=Path(temporary) / "trace.jsonl", max_wall_time_s=10,
            )
            self.assertEqual((Path(temporary) / "lightnav_frames/frame_00001.jpg").read_bytes(), b"after")
        self.assertTrue(result["post_budget_observation_confirmed"])


class ArrivalTests(unittest.IsolatedAsyncioTestCase):
    async def test_requires_metric_proximity_not_only_model_stop(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        depth = np.full((10, 10), 0.8, dtype=np.float32)
        transform = np.eye(4)
        transform[0, 3] = 0.8
        views = tuple(CameraView(
            name=name, image_path=Path(f"{name}.png"), rgb=rgb,
            depth_path=Path(f"{name}.npy"), depth_m=depth,
            intrinsics=np.array([[1000., 0., 4.5], [0., 1000., 4.5], [0., 0., 1.]]),
            T_world_camera=transform,
        ) for name in ("head", "left_wrist"))
        observation = CameraObservation("head", "robot_base", views)

        class FakeStore:
            def __init__(self, _workspace):
                pass

            def load(self):
                return observation

        class FakeSAM3:
            def __init__(self, *_args, **_kwargs):
                pass

            async def infer(self, _request):
                return {"views": [{"view_name": "head", "targets": [{
                    "object_key": "radio", "prompt": "radio receiver", "found": True,
                    "confidence": 0.9, "mask": np.ones((10, 10), dtype=bool),
                }]}]}

        with TemporaryDirectory() as temporary, \
             patch("integrations.lightnav_behavior.arrival.ObservationStore", FakeStore), \
             patch("integrations.lightnav_behavior.arrival.ModelServerClient", FakeSAM3):
            nearby = await verify_radio_handoff(
                Path(temporary), sam3_url="ws://sam3", max_horizontal_distance_m=1.0,
                overlay_path=Path(temporary) / "overlay.png",
            )
            far = await verify_radio_handoff(
                Path(temporary), sam3_url="ws://sam3", max_horizontal_distance_m=0.5,
                overlay_path=Path(temporary) / "overlay2.png",
            )
            self.assertTrue((Path(temporary) / "overlay.png").is_file())
        self.assertTrue(nearby["verified"])
        self.assertFalse(far["verified"])


if __name__ == "__main__":
    unittest.main()
