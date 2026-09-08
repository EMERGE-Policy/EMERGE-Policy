"""Orange-led terminal palette with semantic sidebar colors."""
from prompt_toolkit.styles import Style


def style(light: bool = False) -> Style:
    if light:
        colors = {
            "bg": "#fff8ee", "panel": "#f4e7d4", "text": "#30261d",
            "muted": "#806f60", "accent": "#d96b18", "brand": "#9a4300",
        }
    else:
        colors = {
            "bg": "#110f0d", "panel": "#1d1813", "text": "#f3e8da",
            "muted": "#9d8b78", "accent": "#ff9d3d", "brand": "#ffd09a",
        }
    c = colors
    return Style.from_dict({
        "": f"bg:{c['bg']} {c['text']}",
        "header": f"bg:{c['panel']}",
        "brand.logo": f"{c['accent']} bold",
        "brand.name": f"{c['brand']} bold",
        "brand.meta": c["muted"],
        "footer": f"bg:{c['panel']} {c['muted']}",
        "frame.border": "#6f4c2b",
        "frame.label": c["accent"],
        "input": f"bg:{c['panel']} {c['text']}",
        "user": f"{c['accent']} bold",
        "assistant": f"{c['text']} bold",
        "tool": "#c7a47a",
        "system": c["muted"],
        "palette": f"bg:{c['panel']} {c['text']}",
        "selected": f"bg:{c['accent']} {c['bg']} bold",
        "sidebar": c["muted"],
        "sidebar.label": "#9aa4ad",
        "sidebar.value": "#f2d6ad",
        "sidebar.path": "#d8a7ff",
        "sidebar.count": "#74d3e8 bold",
        "sidebar.good": "#77d68a",
        "sidebar.bad": "#ff6b5f",
        "sidebar.warn": "#ffc266",
        "plan.current": f"bg:{c['accent']} {c['bg']} bold",
        "plan.done": "#77d68a",
        "plan.pending": "#7f8993",
        "plan.progress": f"{c['accent']} bold",
        "robot.name": "#55d6be bold",
        "action.name": "#d8a7ff",
        "observation.reference": "#f29bd4 bold",
        "service.vggt": "#57d7c4 bold",
        "service.sam3": "#7eb6ff bold",
        "sidebar.session": "#65c7d0 bold",
        "sidebar.plan": "#f2c14e bold",
        "sidebar.robot": "#77c66e bold",
        "sidebar.observation": "#d58adf bold",
        "sidebar.services": "#70a7ff bold",
        "sidebar.artifacts": c["accent"] + " bold",
        "status.ready": "#77c66e",
        "status.completed": "#77c66e",
        "status.running": "#70a7ff",
        "status.pending": "#f2c14e",
        "status.failed": "#ff6b5f",
        "status.unavailable": "#ff6b5f",
        "status.cancelled": "#d58adf",
        "status.unknown": c["muted"],
        "error": "#ff6b5f",
    })
