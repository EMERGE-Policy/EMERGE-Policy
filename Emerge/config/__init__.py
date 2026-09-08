"""Configuration module for Emerge."""

from Emerge.config.loader import get_config_path, load_config
from Emerge.config.paths import (
    get_bridge_install_dir,
    get_cli_history_path,
    get_data_dir,
    get_legacy_sessions_dir,
    get_logs_dir,
    get_runtime_subdir,
    get_workspace_path,
)
from Emerge.config.schema import Config

__all__ = [
    "Config",
    "load_config",
    "get_config_path",
    "get_data_dir",
    "get_runtime_subdir",
    "get_logs_dir",
    "get_workspace_path",
    "get_cli_history_path",
    "get_bridge_install_dir",
    "get_legacy_sessions_dir",
]
