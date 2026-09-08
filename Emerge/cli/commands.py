"""Interactive and configuration command dispatch."""
from __future__ import annotations

import argparse
import sys


def app():
    argv = sys.argv[1:]
    if argv and argv[0] in {"onboard", "wsinit", "provider"}:
        from Emerge.cli.management import app as management
        management(args=argv)
        return

    parser = argparse.ArgumentParser(
        prog="emerge", description="Emerge robot agent workspace",
        epilog="Commands: onboard, wsinit, provider",
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
    from Emerge.tui.app import launch
    launch(config=args.config, workspace=args.workspace, session_id=args.session, model=args.model)


if __name__ == "__main__":
    app()
