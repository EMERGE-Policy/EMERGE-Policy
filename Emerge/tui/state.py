"""Presentation-only reduction of Runtime events."""
from dataclasses import dataclass, field


@dataclass
class ViewState:
    session_id: str
    model: str
    status: str = "idle"
    entries: list[dict] = field(default_factory=list)
    messages: dict[str, dict] = field(default_factory=dict)
    tools: dict[str, dict] = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    duration_ms: int = 0

    def note(self, text):
        self.entries.append({"role": "system", "text": text})

    def accept(self, event):
        data = event.data
        if event.type == "run.started":
            self.status = "starting"
            self.entries.append({"role": "user", "text": data["message"]})
        elif event.type == "run.state":
            self.status = data["state"]
        elif event.type == "run.configured":
            self.model = data["model"]
        elif event.type in {"assistant.delta", "assistant.message"}:
            key = data["message_id"]
            if key not in self.messages:
                self.messages[key] = {"role": "assistant", "text": ""}
                self.entries.append(self.messages[key])
            entry = self.messages[key]
            entry["text"] = entry["text"] + data["text"] if event.type.endswith("delta") else data["text"]
            entry["streaming"] = event.type.endswith("delta")
        elif event.type.startswith("tool."):
            key = event.run_id + ":" + data["tool_call_id"]
            if key not in self.tools:
                self.tools[key] = {"role": "tool"}
                self.entries.append(self.tools[key])
            self.tools[key].update(data, status=event.type.split(".")[1])
        elif event.type == "run.finished":
            self.status = data["run_status"]
            self.usage = data["usage"]
            self.duration_ms = data["duration_ms"]
            if data["error"]:
                self.note(data["error"]["message"])
            self.note(data["run_status"] + " · " + data["finish_reason"])
