"""Project runtime events into browser messages without depending on the TUI."""

import json

from Emerge.runtime.protocol import RunEvent


class RunView:
    def __init__(self):
        self.messages = []
        self.by_id = {}
        self.usage = {}
        self.duration_ms = 0
        self.verification = None
        self.result = None

    def accept(self, event: RunEvent):
        data, kind = event.data, event.type
        if kind == "run.started":
            self.verification = None
            self.result = None
            self.messages.append(
                {"id": event.run_id + ":user", "role": "user", "text": data["message"]}
            )
        elif kind in {"assistant.delta", "assistant.message"}:
            key = data["message_id"]
            if key not in self.by_id:
                self.by_id[key] = {"id": key, "role": "assistant", "text": ""}
                self.messages.append(self.by_id[key])
            message = self.by_id[key]
            message["text"] = (
                message["text"] + data["text"] if kind == "assistant.delta" else data["text"]
            )
        elif kind.startswith("tool."):
            key = event.run_id + ":" + data["tool_call_id"]
            if key not in self.by_id:
                self.by_id[key] = {"id": key, "role": "tool", "text": ""}
                self.messages.append(self.by_id[key])
            message = self.by_id[key]
            message.update(
                tool=data["tool"],
                status="running" if kind == "tool.started" else kind.split(".")[1],
            )
            if "arguments" in data:
                message["arguments"] = json.dumps(data["arguments"], ensure_ascii=False, indent=2)
            if "result" in data:
                message["text"] = data["result"]
                if data["tool"] == "delegate_subagent":
                    try:
                        result = json.loads(data["result"])
                    except ValueError:
                        result = {}
                    if result.get("agent_name") == "task_verification" and isinstance(
                        result.get("output"), dict
                    ):
                        self.verification = result["output"]
                        message["verification"] = self.verification
            if "duration_ms" in data:
                message["duration"] = f"{data['duration_ms'] / 1000:.2f}s"
        elif kind == "run.finished":
            self.result = data
            self.usage = data.get("usage", {})
            self.duration_ms = data.get("duration_ms", 0)
            detail = data["run_status"] + " · " + data["finish_reason"]
            if data.get("error"):
                detail += ": " + data["error"]["message"]
            self.messages.append({"id": event.run_id + ":result", "role": "system", "text": detail})
        elif kind == "agent.progress" and not data.get("tool_hint"):
            # The loop repeats assistant text as progress. Keep only standalone updates.
            if not any(
                message["role"] == "assistant" and message["text"] == data["text"]
                for message in self.messages
            ):
                self.messages.append(
                    {
                        "id": f"{event.run_id}:progress:{event.seq}",
                        "role": "assistant",
                        "text": data["text"],
                    }
                )
