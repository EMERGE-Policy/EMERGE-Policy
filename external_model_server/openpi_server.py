"""Single-port OpenPI model service."""

import argparse
import logging

from external_model_server.openpi_adapter import OpenPIAdapter
from external_model_server.model_service.runtime import (
    ModelServerRuntime,
    add_runtime_arguments,
    runtime_arguments,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified single-port OpenPI model service")
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--batch-wait-ms", type=float, default=0)
    parser.add_argument("--batch-pad-to-max", action="store_true")
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    if args.max_batch_size < 1:
        parser.error("--max-batch-size must be positive")
    if args.batch_wait_ms < 0:
        parser.error("--batch-wait-ms cannot be negative")
    adapter = OpenPIAdapter(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        model_id=args.model_id,
        max_batch_size=args.max_batch_size,
        pad_to_max=args.batch_pad_to_max,
    )
    ModelServerRuntime(
        adapter,
        host=args.host,
        port=args.port,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        **runtime_arguments(args),
    ).serve_forever()


if __name__ == "__main__":
    main()
