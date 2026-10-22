"""Check a real official Robometer server with saved RGB LIBERO frames.

Run with EmergePolicy Python, from the repository root. No model imports.
Each --frames must be a uint8 (2,H,W,3) .npy of two ordered frames from ONE camera.
This tests the interface/model only, not Cosmos prediction or robot execution.
"""
import argparse
import json
import numpy as np

from external_model_server.wam.robometer import DEFAULT_MODEL_PATH, RobometerClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8100")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--task", required=True)
    parser.add_argument("--frames", action="append", required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    sequences = [np.load(path, allow_pickle=False) for path in args.frames]
    if any(x.ndim != 4 or x.shape[0] != 2 for x in sequences):
        parser.error("Each --frames must have layout (2,H,W,3)")
    if any(not np.array_equal(x[0], sequences[0][0]) for x in sequences[1:]):
        parser.error("Candidates must share the same current frame")
    client = RobometerClient(args.url, model_path=args.model_path, timeout=args.timeout)
    try:
        scores, info = client.score(sequences[0][0], [x[1] for x in sequences],
                                    args.task, list(range(len(sequences))))
        print(json.dumps({"scores": scores, "scoring": info}, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
