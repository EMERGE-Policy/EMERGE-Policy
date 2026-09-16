"""Emerge-side WAM protocol, planning, and MuJoCo execution."""

__all__ = ["CosmosWAMClient", "CosmosWAMExecutor", "WAMResult"]


def __getattr__(name: str):
    """Keep protocol-only imports from loading the client and executor."""
    if name == "CosmosWAMClient":
        from .client import CosmosWAMClient

        return CosmosWAMClient
    if name in {"CosmosWAMExecutor", "WAMResult"}:
        from .mujoco_policy_executor import CosmosWAMExecutor, WAMResult

        return {"CosmosWAMExecutor": CosmosWAMExecutor, "WAMResult": WAMResult}[name]
    raise AttributeError(name)
