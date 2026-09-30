"""First-run configuration shared by the terminal and browser interfaces."""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, ValidationError

from Emerge.config.loader import get_config_path
from Emerge.config.schema import Base, Config
from Emerge.providers.registry import PROVIDERS, find_by_model, find_by_name
from Emerge.runtime.storage import atomic_json


class SetupField(Base):
    name: Literal["model", "api_base", "api_key"]
    title: str
    description: str
    default: str = ""
    required: bool = False
    password: bool = False


class SetupProvider(Base):
    name: str
    label: str
    category: str
    is_oauth: bool
    fields: list[SetupField]


class SetupStatus(Base):
    required: bool
    provider: str
    providers: list[SetupProvider]


class SetupValues(Base):
    provider: str
    model: str
    api_base: str = ""
    api_key: SecretStr = SecretStr("")


def load_setup(config: str | Path | None = None) -> tuple[Path, Config]:
    """Read configuration strictly, allowing a missing file on first run."""
    path = (Path(config) if config else get_config_path()).expanduser().resolve()
    try:
        settings = (
            Config.model_validate_json(path.read_text(encoding="utf-8"))
            if path.exists()
            else Config()
        )
    except ValidationError:
        raise ValueError(f"Invalid config: {path}. Fix the JSON and run emerge again.") from None
    return path, settings


def describe_setup(settings: Config, path: Path, model: str | None = None) -> SetupStatus:
    """Describe readiness and input fields without returning saved credentials."""
    selected_model = model or settings.agents.defaults.model
    spec = find_by_name(settings.get_provider_name(selected_model))
    provider = settings.get_provider(selected_model)
    ready = bool(
        spec
        and (spec.is_oauth or spec.is_local or spec.name == "custom" or provider.api_key.strip())
    )
    if spec and spec.name in {"azure_openai", "vllm"}:
        ready = ready and bool(provider.api_base)
    if selected_model.startswith("bedrock/") and settings.agents.defaults.provider == "auto":
        ready = True  # Bedrock uses the AWS credential chain.

    suggested = spec or find_by_model(selected_model)
    preferred = ["openai", "anthropic", "gemini", "openrouter", "custom"]
    choices = sorted(
        PROVIDERS,
        key=lambda item: preferred.index(item.name) if item.name in preferred else len(preferred),
    )
    providers = []
    for item in choices:
        provider = getattr(settings.providers, item.name)
        fields = [
            SetupField(
                name="model",
                title="Model",
                description="Enter your deployment name."
                if item.name == "azure_openai"
                else "Use a model with image input and tool calling.",
                default=selected_model if suggested and suggested.name == item.name else "",
                required=True,
            )
        ]
        if not item.is_oauth:
            base = provider.api_base or item.default_api_base
            if item.name in {"custom", "vllm"} and not base:
                base = "http://localhost:8000/v1"
            required_base = item.name in {"azure_openai", "custom", "vllm", "ollama"}
            optional_key = item.is_local or item.name == "custom"
            fields.extend(
                [
                    SetupField(
                        name="api_base",
                        title="API address",
                        description="Enter your API base URL."
                        if required_base
                        else "Leave blank to use the provider's default address.",
                        default=base,
                        required=required_base,
                    ),
                    SetupField(
                        name="api_key",
                        title="API key",
                        description="Leave blank to keep your existing key. Input is hidden."
                        if provider.api_key.strip()
                        else "Optional for your server. Input is hidden."
                        if optional_key
                        else "Paste your API key. Input is hidden.",
                        required=not optional_key and not provider.api_key.strip(),
                        password=True,
                    ),
                ]
            )
        providers.append(
            SetupProvider(
                name=item.name,
                label=item.label,
                category="OAUTH / BROWSER"
                if item.is_oauth
                else "LOCAL RUNTIME"
                if item.is_local
                else "MODEL GATEWAY"
                if item.is_gateway
                else "DIRECT ENDPOINT"
                if item.is_direct
                else "HOSTED API",
                is_oauth=item.is_oauth,
                fields=fields,
            )
        )
    return SetupStatus(
        required=not path.exists() or not selected_model.strip() or not ready,
        provider=suggested.name if suggested else "openai",
        providers=providers,
    )


def complete_setup(
    settings: Config,
    path: Path,
    values: SetupValues,
    *,
    workspace: str | None = None,
) -> Config:
    """Validate and atomically save a completed form, keeping unrelated settings."""
    status = describe_setup(settings, path)
    provider = next((item for item in status.providers if item.name == values.provider), None)
    if provider is None:
        raise ValueError("Choose a provider from the list")
    answers = {
        "model": values.model.strip(),
        "api_base": values.api_base.strip(),
        "api_key": values.api_key.get_secret_value().strip(),
    }
    for field in provider.fields:
        if field.required and not answers[field.name]:
            raise ValueError(f"{field.title} is required for {provider.label}")

    updated = settings.model_copy(deep=True)
    updated.agents.defaults.provider = provider.name
    updated.agents.defaults.model = answers["model"]
    if not provider.is_oauth:
        target = getattr(updated.providers, provider.name)
        target.api_base = answers["api_base"] or None
        if answers["api_key"]:
            target.api_key = answers["api_key"]
    if workspace:
        updated.agents.defaults.workspace = str(Path(workspace).expanduser().resolve())
    atomic_json(path, updated.model_dump(by_alias=True))
    return updated
