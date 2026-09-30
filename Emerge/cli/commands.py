"""Interactive and configuration command dispatch."""
from __future__ import annotations

import argparse
import sys


def app():
    argv = sys.argv[1:]
    if argv and argv[0] == "web":
        from Emerge.web.app import main
        main(argv[1:])
        return
    if argv and argv[0] in {"workspace", "provider"}:
        from Emerge.cli.management import app as management
        management(args=argv)
        return

    parser = argparse.ArgumentParser(
        prog="emerge", description="Emerge robot agent workspace",
        epilog="Commands: workspace, provider",
    )
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--workspace", "-w")
    parser.add_argument("--config", "-c")
    parser.add_argument("--session", "-s")
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    if args.version:
        from Emerge import __version__
        print(__version__)
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("Interactive mode needs a terminal")
    from click.exceptions import Abort, Exit

    from Emerge.tui.setup import prepare_config

    try:
        config_path = prepare_config(config=args.config, workspace=args.workspace, model=args.model)
    except (KeyboardInterrupt, EOFError, Abort):
        return
    except Exit as exc:
        raise SystemExit(exc.exit_code) from None
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    if config_path is None:
        return
    from Emerge.tui.app import launch
    launch(config=str(config_path), workspace=args.workspace, session_id=args.session, model=args.model)


if __name__ == "__main__":
    app()
