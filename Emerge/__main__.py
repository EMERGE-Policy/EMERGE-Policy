"""
Entry point for running Emerge as a module: python -m Emerge
"""

from Emerge.cli.commands import app

if __name__ == "__main__":
    app()
