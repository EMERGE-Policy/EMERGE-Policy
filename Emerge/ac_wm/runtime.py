"""Build the optional Cosmos3 rollout and independent VLM judge."""

import os


def build_components(config):
    enabled = config.ac_wm.enabled or os.environ.get("EMERGE_AC_WM", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if not enabled:
        return None, None

    from .cosmos_adapter import CosmosFrameworkAdapter, CosmosFrameworkConfig
    from .vlm_judge import OpenAICompatibleVlmJudge

    adapter = CosmosFrameworkAdapter(
        CosmosFrameworkConfig(
            python=os.environ.get("COSMOS_PYTHON", ""),
            checkpoint=os.environ.get("COSMOS_CHECKPOINT", "Cosmos3-Edge"),
            config_file=os.environ.get("COSMOS_CONFIG") or None,
            server_url=os.environ.get("COSMOS_SERVER_URL") or None,
            action_horizon=int(os.environ.get("COSMOS_ACTION_HORIZON", "52")),
            libero_action_stats_path=os.environ.get("COSMOS_LIBERO_ACTION_STATS"),
        )
    )

    def rollout(request, candidate):
        return adapter.run(request, candidate, request.output_dir)

    rollout.run_many = lambda request, candidates: adapter.run_many(
        request,
        candidates,
        request.output_dir,
    )
    return rollout, OpenAICompatibleVlmJudge.from_config(config)
