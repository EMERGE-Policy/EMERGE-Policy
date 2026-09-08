"""Provider-neutral content blocks used by multimodal sub-agents."""

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class TextContent:
    """A text block in a sub-agent message."""

    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Text content cannot be empty")

    def to_message_part(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True, slots=True)
class ImageContent:
    """An image URL or data URL in a sub-agent message."""

    image_url: str
    detail: Literal["auto", "low", "high"] = "auto"

    def __post_init__(self) -> None:
        if not self.image_url.strip():
            raise ValueError("Image URL cannot be empty")

    def to_message_part(self) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": self.image_url, "detail": self.detail},
        }


ContentPart: TypeAlias = TextContent | ImageContent


def to_message_content(content: tuple[ContentPart, ...]) -> list[dict[str, Any]]:
    """Convert framework content blocks to canonical model message parts."""
    return [part.to_message_part() for part in content]
