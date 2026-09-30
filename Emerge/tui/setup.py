"""Interactive first-run setup before opening the chat workspace."""

from pathlib import Path

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    ConditionalContainer,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import TextArea

from Emerge.config.setup import SetupValues, complete_setup, describe_setup, load_setup
from Emerge.tui.theme import style
from Emerge.utils.helpers import sync_workspace_templates

Option = tuple[str, str, str]
WIDE_LAYOUT_COLUMNS = 132
WIDE_RAIL_COLUMNS = 36
MAIN_PADDING_COLUMNS = 8
FIXED_CONTENT_ROWS = 13
SETUP_BRAND = (
    "█▀▀ █▄█ █▀▀ █▀█ █▀▀ █▀▀",
    "█▀▀ █ █ █▀▀ ██▀ █▄█ █▀▀",
    "▀▀▀ ▀ ▀ ▀▀▀ ▀ ▀ ▀▀▀ ▀▀▀",
)


def _prompt(
    title: str,
    description: str,
    *,
    step: int = 1,
    options: list[Option] | None = None,
    default: str = "",
    password: bool = False,
    required: bool = False,
) -> str | None:
    """Render one responsive full-screen setup step."""
    keys = KeyBindings()
    error = ""
    selected = next((i for i, item in enumerate(options or []) if item[0] == default), 0)
    field = TextArea(
        text=default if not options else "",
        multiline=False,
        password=password,
        prompt=[("class:setup.input-marker", "  > ")],
        height=1,
        wrap_lines=False,
        style="class:setup.input",
    )
    field.buffer.cursor_position = len(field.text)

    def is_wide() -> bool:
        return get_app().output.get_size().columns >= WIDE_LAYOUT_COLUMNS

    def content_columns() -> int:
        columns = get_app().output.get_size().columns
        reserved = WIDE_RAIL_COLUMNS if is_wide() else 0
        return max(52, columns - reserved - MAIN_PADDING_COLUMNS)

    def visible_range() -> tuple[int, int]:
        available = max(5, get_app().output.get_size().rows - FIXED_CONTENT_ROWS)
        count = min(available, len(options or []))
        start = min(max(selected - count // 2, 0), max(len(options or []) - count, 0))
        return start, count

    def option_cell(index: int, width: int):
        if index >= len(options):
            return [("", " " * width)]
        _, label, detail = options[index]
        detail_width = min(22, max(14, len(detail)))
        label_width = max(12, width - detail_width - 5)
        prefix = " > " if index == selected else "   "
        label_text = label[:label_width]
        if index == selected:
            text = f"{prefix}{label_text:<{label_width}}{detail:>{detail_width}}  "
            return [("class:setup.selected", text[:width].ljust(width))]
        return [
            ("class:setup.option", f"{prefix}{label_text:<{label_width}}"),
            ("class:setup.option-meta", f"{detail:>{detail_width}}  "),
        ]

    def menu():
        start, count = visible_range()
        lines = []
        for i in range(start, start + count):
            lines.extend(option_cell(i, content_columns()))
            if i < start + count - 1:
                lines.append(("", "\n"))
        return lines

    def heading():
        width = content_columns()
        left = "FIRST-TIME SETUP"
        right = f"STEP {step:02d}"
        return [
            ("class:setup.eyebrow", left),
            ("class:setup.step", " " * max(3, width - len(left) - len(right)) + right),
        ]

    def footer():
        if not options:
            return [
                ("class:setup.key", "Enter"),
                ("class:setup.help", " Continue    "),
                ("class:setup.key", "Esc"),
                ("class:setup.help", " Cancel"),
            ]
        start, count = visible_range()
        scroll = ""
        if start:
            scroll += "  ↑ more"
        if start + count < len(options):
            scroll += "  ↓ more"
        return [
            ("class:setup.key", "↑↓"),
            ("class:setup.help", " Navigate    "),
            ("class:setup.key", "Enter"),
            ("class:setup.help", " Select    "),
            ("class:setup.key", "Esc"),
            ("class:setup.help", " Cancel"),
            ("class:setup.position", f"    {selected + 1:02d} / {len(options):02d}{scroll}"),
        ]

    if options:

        @keys.add("up")
        @keys.add("down")
        def move(event):
            nonlocal selected
            direction = 1 if event.key_sequence[0].key == "down" else -1
            candidate = selected + direction
            if 0 <= candidate < len(options):
                selected = candidate

        @keys.add("pageup")
        @keys.add("pagedown")
        def move_page(event):
            nonlocal selected
            direction = 1 if event.key_sequence[0].key == "pagedown" else -1
            _, page_size = visible_range()
            selected = min(max(selected + direction * page_size, 0), len(options) - 1)

        @keys.add("home")
        def move_first(event):
            nonlocal selected
            selected = 0

        @keys.add("end")
        def move_last(event):
            nonlocal selected
            selected = len(options) - 1

    @keys.add("enter")
    def accept(event):
        nonlocal error
        value = options[selected][0] if options else field.text.strip()
        if required and not value:
            error = "Please enter a value."
        else:
            event.app.exit(result=value)

    @keys.add("c-c")
    @keys.add("escape")
    def cancel(event):
        event.app.exit(result=None)

    if options:
        body = Window(
            FormattedTextControl(menu, focusable=True),
            height=Dimension(min=min(5, len(options)), preferred=len(options), max=len(options)),
            always_hide_cursor=True,
        )
    else:
        body = field

    input_area = (
        body
        if options
        else HSplit(
            [
                Window(height=1, style="class:setup.input"),
                field,
                Window(height=1, style="class:setup.input"),
            ]
        )
    )
    content = HSplit(
        [
            Window(height=2),
            Window(
                FormattedTextControl(heading),
                height=1,
            ),
            Window(height=1, char="─", style="class:setup.rule"),
            Window(height=1),
            Window(FormattedTextControl([("class:setup.title", title.upper())]), height=1),
            Window(
                FormattedTextControl([("class:setup.description", description)]),
                height=2,
                wrap_lines=True,
            ),
            Window(),
            input_area,
            Window(),
            Window(
                FormattedTextControl(lambda: [("class:error", f"  {error}" if error else "")]),
                height=1,
            ),
            Window(height=1),
            Window(FormattedTextControl(footer), height=1),
            Window(height=2),
        ],
    )

    rail_content = HSplit(
        [
            Window(height=1),
            Window(
                FormattedTextControl([("class:setup.rail-logo", "\n".join(SETUP_BRAND))]),
                height=3,
            ),
            Window(height=1),
            Window(FormattedTextControl([("class:setup.rail-meta", "WORKSPACE / FIRST RUN")]), height=1),
            Window(height=2),
            Window(FormattedTextControl([("class:setup.rail-meta", "CURRENT STEP")]), height=1),
            Window(FormattedTextControl([("class:setup.rail-step", f"{step:02d}  /  SETUP")]), height=1),
            Window(height=1),
            Window(FormattedTextControl([("class:setup.rail-title", title.upper())]), wrap_lines=True),
            Window(),
            Window(
                FormattedTextControl(
                    [
                        (
                            "class:setup.rail-meta",
                            "Saved locally when\nsetup completes.",
                        ),
                    ]
                ),
                height=2,
            ),
            Window(height=1),
        ],
        width=28,
        style="class:setup.rail",
    )
    rail = VSplit(
        [
            Window(width=2, style="class:setup.rail"),
            rail_content,
            Window(width=2, style="class:setup.rail"),
        ],
        width=32,
        style="class:setup.rail",
    )
    wide = Condition(lambda: get_app().output.get_size().columns >= WIDE_LAYOUT_COLUMNS)
    main = VSplit(
        [
            Window(width=4),
            content,
            Window(width=4),
        ],
    )
    root = VSplit(
        [
            ConditionalContainer(rail, filter=wide),
            ConditionalContainer(Window(width=4), filter=wide),
            main,
        ],
        style="class:background",
    )
    return Application(
        layout=Layout(root, focused_element=body),
        key_bindings=keys,
        style=style(),
        full_screen=True,
        mouse_support=False,
    ).run()


def prepare_config(
    *,
    config: str | None = None,
    workspace: str | None = None,
    model: str | None = None,
) -> Path | None:
    """Configure missing credentials, save on completion, and create missing templates."""
    path, settings = load_setup(config)
    status = describe_setup(settings, path, model)
    if status.required:
        name = _prompt(
            "Choose your provider",
            "Connect the model service used by this workspace.",
            options=[(item.name, item.label, item.category) for item in status.providers],
            default=status.provider,
        )
        if name is None:
            return None
        provider = next(item for item in status.providers if item.name == name)
        answers = {"provider": name}
        if model:
            answers["model"] = model
        fields = [field for field in provider.fields if field.name != "model" or not model]
        for step, field in enumerate(fields, 2):
            value = _prompt(
                f"{field.title} · {provider.label}",
                field.description,
                step=step,
                default=field.default,
                password=field.password,
                required=field.required,
            )
            if value is None:
                return None
            answers[field.name] = value

        if provider.is_oauth:
            from Emerge.cli.management import provider_login

            provider_login(name)
        settings = complete_setup(settings, path, SetupValues(**answers), workspace=workspace)

    workspace_path = (
        Path(workspace).expanduser().resolve() if workspace else settings.workspace_path
    )
    sync_workspace_templates(workspace_path, silent=True)
    return path
