#!/usr/bin/env python3
"""Script to set up git hooks for development."""

import subprocess
import sys
from pathlib import Path


def install_pre_commit() -> int:
    """Install pre-commit and configure hooks."""
    try:
        # Install pre-commit if not already installed
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pre-commit"],
            check=True
        )

        # Install the git hooks
        subprocess.run(
            ["pre-commit", "install"],
            check=True
        )

        print("\nPre-commit hooks installed successfully!")
        print("\nThe following checks will run before each commit:")
        print("1. Non-integration tests")
        print("2. Security scanning")
        print("\nTo bypass hooks temporarily, use: git commit --no-verify")

        return 0

    except subprocess.CalledProcessError as e:
        print(f"Error setting up pre-commit hooks: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(install_pre_commit())
