"""Offline checks for the read-only, multi-frame BEHAVIOR LightNav preview."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from PIL import Image

from integrations.lightnav_behavior.preview import LightNavPreviewTool
from scripts.Behavior_eval.runner_common import atomic_write_json


def publish(workspace: Path, *, episode: int, seq: int, revision: int) -> None:
    image_path = workspace / f"artifacts/cameras/head/head.{revision:06d}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (220, 30, 20)).save(image_path)
    atomic_write_json(workspace / "artifacts/observations/observation.json", {
        "episode_id": episode,
        "observation_seq": seq,
        "revision": revision,
        "views": [{
            "name": "head",
            "image_path": str(image_path.relative_to(workspace)),
            "width": 16,
            "height": 16,
        }],
    })


class FakeClient:
    def __init__(self, url: str, *, client_id: str, stop: bool = False) -> None:
        self.url = url
        self.client_id = client_id
        self.stop = stop
        self.requests: list[tuple[bytes, str]] = []
        self.closed = False

    def predict(self, jpeg: bytes, instruction: str):
        self.requests.append((jpeg, instruction))
        return SimpleNamespace(first_waypoint=(0.1, -0.02, 0.03), stop=self.stop, latency_ms=12.0)

    def close(self):
        self.closed = True


class PreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_samples_fresh_frames_and_keeps_session_across_calls(self):
        with TemporaryDirectory() as temp:
            workspace = Path(temp)
            atomic_write_json(workspace / "SHADOW_STATUS.json", {"phase": "running"})
            publish(workspace, episode=1, seq=1, revision=1)
            clients: list[FakeClient] = []

            def factory(url, *, client_id):
                client = FakeClient(url, client_id=client_id)
                clients.append(client)
                return client

            tool = LightNavPreviewTool(workspace, client_factory=factory)

            async def publish_next():
                await asyncio.sleep(0.25)
                publish(workspace, episode=1, seq=10, revision=2)
                await asyncio.sleep(0.25)
                publish(workspace, episode=1, seq=20, revision=3)

            publisher = asyncio.create_task(publish_next())
            first = json.loads(await asyncio.wait_for(
                tool.execute("Approach the red radio", max_frames=2, timeout_s=3), timeout=5,
            ))
            second = json.loads(await asyncio.wait_for(
                tool.execute("Approach the red radio", max_frames=1, timeout_s=3), timeout=5,
            ))
            await publisher
            self.assertEqual([item["observation_seq"] for item in first["predictions"]], [1, 10])
            self.assertEqual(second["predictions"][0]["observation_seq"], 20)
            self.assertEqual(len(clients), 1)
            self.assertEqual(len(clients[0].requests), 3)
            self.assertTrue(all(request[0].startswith(b"\xff\xd8") for request in clients[0].requests))
            log = workspace / "artifacts/lightnav_preview/predictions.jsonl"
            self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 3)
            self.assertEqual(len(list((workspace / "artifacts/lightnav_preview/frames").glob("*.jpg"))), 3)

            publish(workspace, episode=2, seq=1, revision=4)
            third = json.loads(await tool.execute("Approach the red radio", max_frames=1, timeout_s=2))
            self.assertEqual(third["predictions"][0]["episode_id"], 2)
            self.assertTrue(clients[0].closed)
            self.assertEqual(len(clients), 2)
            tool.close()
            self.assertTrue(clients[1].closed)

    async def test_stop_is_only_a_prediction_and_does_not_actuate(self):
        with TemporaryDirectory() as temp:
            workspace = Path(temp)
            atomic_write_json(workspace / "SHADOW_STATUS.json", {"phase": "running"})
            publish(workspace, episode=1, seq=1, revision=1)
            tool = LightNavPreviewTool(
                workspace, client_factory=lambda url, *, client_id: FakeClient(url, client_id=client_id, stop=True)
            )
            result = json.loads(await tool.execute("Approach the red radio", max_frames=3, timeout_s=2))
            self.assertEqual(result["sample_count"], 1)
            self.assertEqual(result["finish_reason"], "model_suggested_stop")
            self.assertEqual(result["mode"], "read_only")
            self.assertFalse((workspace / "ACTION.md").exists())
            tool.close()

    async def test_completed_episode_does_not_query_lightnav(self):
        with TemporaryDirectory() as temp:
            workspace = Path(temp)
            atomic_write_json(workspace / "SHADOW_STATUS.json", {"phase": "completed"})
            publish(workspace, episode=1, seq=1, revision=1)
            clients = []
            tool = LightNavPreviewTool(
                workspace, client_factory=lambda url, *, client_id: clients.append((url, client_id))
            )
            result = json.loads(await tool.execute("Approach the red radio"))
            self.assertEqual(result["finish_reason"], "episode_finished")
            self.assertEqual(clients, [])

    async def test_rejects_unbounded_requests(self):
        with TemporaryDirectory() as temp:
            tool = LightNavPreviewTool(Path(temp))
            with self.assertRaisesRegex(ValueError, "max_frames"):
                await tool.execute("Approach radio", max_frames=100)
            with self.assertRaisesRegex(ValueError, "instruction"):
                await tool.execute(" ")


if __name__ == "__main__":
    unittest.main()
