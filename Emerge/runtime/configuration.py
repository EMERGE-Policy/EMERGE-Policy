"""Strict configuration loading for execution; no stdout or global config mutation."""

import json
from pathlib import Path

from Emerge.config.loader import get_config_path
from Emerge.config.schema import Config
from Emerge.runtime.protocol import RunRequest


def load_runtime_config(request: RunRequest) -> Config:
    path = Path(request.config).expanduser().resolve() if request.config else get_config_path()
    if request.config and not path.is_file():
        raise ValueError(f"Config file not found: {path}")
    config = Config.model_validate(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else Config()
    if request.workspace:
        config.agents.defaults.workspace = str(Path(request.workspace).expanduser().resolve())
    if request.model:
        config.agents.defaults.model = request.model
    if request.max_iterations is not None:
        config.agents.defaults.max_tool_iterations = request.max_iterations
    if request.restrict_to_workspace is not None:
        config.tools.restrict_to_workspace = request.restrict_to_workspace
    return config
