import json

import cv2
import numpy as np

from Emerge.ac_wm.protocol import ActionCandidate, RolloutRequest, RolloutResult
from Emerge.ac_wm.vlm_judge import OpenAICompatibleVlmJudge, VlmJudgeConfig


def test_joint_judge_decodes_futures_and_includes_dispatch_endpoint(tmp_path, monkeypatch):
    observed = tmp_path / "observed.png"
    assert cv2.imwrite(str(observed), np.zeros((32, 32, 3), dtype=np.uint8))
    rollouts = {}
    candidates = []
    for name, delta in [("left", -0.2), ("right", 0.2)]:
        video = tmp_path / f"{name}.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20, (32, 32))
        assert writer.isOpened()
        for index in range(9):
            writer.write(np.full((32, 32, 3), index * 20, dtype=np.uint8))
        writer.release()
        candidates.append(
            ActionCandidate(
                name, "rule:move_linear", ((delta, 0, 0, 0, 0, 0, -1),) * 8, {"execute_steps": 3}
            )
        )
        rollouts[name] = RolloutResult(name, "success", str(video), {"action_horizon": 8})
    request = RolloutRequest("recover grasp", str(observed), tuple(candidates), "libero")
    bodies = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            scores = {
                "candidates": [
                    {"candidate_id": "left", "score": 0.2, "rationale": "small progress"},
                    {"candidate_id": "right", "score": 0.7, "rationale": "better progress"},
                ]
            }
            return json.dumps({"choices": [{"message": {"content": json.dumps(scores)}}]}).encode()

    def respond(req, timeout):
        bodies.append(json.loads(req.data))
        return Response()

    monkeypatch.setattr("Emerge.ac_wm.vlm_judge.urlopen", respond)
    judge = OpenAICompatibleVlmJudge(
        VlmJudgeConfig("https://example.test/v1", "test-vision", api_key_env="", api_key="test")
    )
    scores = judge.evaluate_candidates(request, rollouts)
    assert scores["right"][0] == 0.7
    assert len(bodies) == 1
    assert bodies[0]["model"] == "test-vision"
    for sequence in judge.last_audit["frame_sequences"].values():
        assert 3 in [frame["action_step"] for frame in sequence]
        assert sequence[-1]["action_step"] == 8
    content = bodies[0]["messages"][0]["content"]
    assert sum(item["type"] == "image_url" for item in content) > 3
