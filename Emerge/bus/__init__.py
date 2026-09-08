"""Message bus module for decoupled channel-agent communication."""

from Emerge.bus.events import InboundMessage, OutboundMessage
from Emerge.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
