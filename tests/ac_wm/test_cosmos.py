import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from Emerge.ac_wm.cosmos_adapter import CosmosFrameworkAdapter, CosmosFrameworkConfig
from Emerge.ac_wm.protocol import ActionCandidate, RolloutRequest
from external_model_server.cosmos3_server import Cosmos3Adapter
from external_model_server.model_service.runtime import ModelServerRuntime


def make_adapter(tmp_path, **kwargs):
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"global_raw": {"q01": [-1] * 9 + [0], "q99": [1] * 10}}))
    return CosmosFrameworkAdapter(
        CosmosFrameworkConfig(
            python="", action_horizon=8, libero_action_stats_path=str(stats), **kwargs
        )
    )


def make_request(tmp_path, count=1):
    candidates = tuple(
        ActionCandidate(
            f"candidate-{i}",
            "vla",
            ((0.5, 0, 0, 0, 0, 0, -1 if i == 0 else 1),) * 6,
            {"prediction_steps": 6, "execute_steps": 3},
        )
        for i in range(count)
    )
    return RolloutRequest(
        "pick up the bowl\nCompare recovery choices",
        str(tmp_path / "observation.png"),
        candidates,
        "libero",
        view_point="concat_view",
        output_dir=str(tmp_path),
    )


def test_cosmos_input_preserves_controller_units_and_real_horizon(tmp_path):
    adapter = make_adapter(tmp_path)
    req = make_request(tmp_path, 2)
    for item, gripper in zip(req.candidates, (1, -1)):
        spec_path = adapter.write_input_spec(req, item, str(tmp_path / f"{item.candidate_id}.json"))
        spec = json.loads(Path(spec_path).read_text())
        actions = np.array(json.loads(Path(spec["action_path"]).read_text()))
        assert spec["model_mode"] == "forward_dynamics"
        assert spec["action_chunk_size"] == 4
        assert spec["num_frames"] == 5
        assert "Compare recovery" not in spec["prompt"]
        assert actions.shape == (4, 10)
        np.testing.assert_allclose(actions[0], [0.5, 0, 0, 1, 0, 0, 0, 1, 0, gripper])


def test_persistent_cosmos_uses_the_main_model_service_protocol(tmp_path):
    class Sample:
        def __init__(self, path):
            self.spec = json.loads(path.read_text())
            self.name = self.spec["name"]

        def download(self, path):
            path.mkdir(parents=True)

        def build_sample(self, **kwargs):
            return self

    class Overrides:
        @classmethod
        def from_files(cls, paths, **kwargs):
            return [Sample(path) for path in paths]

    class Pipeline:
        model_config = {}

        def generate(self, samples):
            for sample in samples:
                (sample.output_dir / "vision.mp4").write_bytes(b"test prediction")
            return [SimpleNamespace(status="success") for _ in samples]

    service = Cosmos3Adapter({"checkpoint_path": "test-cosmos3"})
    service.setup = SimpleNamespace(get_sample_overrides_cls=lambda: Overrides, sample_overrides={})
    service.pipeline = Pipeline()
    service.load = lambda: None

    async def exercise():
        runtime = ModelServerRuntime(service, port=0)
        serving = asyncio.create_task(runtime.run())
        try:
            await runtime.bound.wait()

            async def ready():
                while runtime.status == "starting":
                    await asyncio.sleep(0.01)
                assert runtime.status == "ready"

            await asyncio.wait_for(ready(), 5)
            adapter = make_adapter(tmp_path, server_url=f"ws://127.0.0.1:{runtime.port}")
            req = make_request(tmp_path, 2)
            result = await asyncio.to_thread(adapter.run_many, req, req.candidates, str(tmp_path))
            assert list(result) == ["candidate-0", "candidate-1"]
            assert all(item.status == "success" for item in result.values())
            assert all(item.metadata["action_horizon"] == 4 for item in result.values())
            assert service._request_count == 1
        finally:
            runtime.stop()
            await serving

    asyncio.run(exercise())
