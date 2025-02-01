
"""Script to bump version numbers across the project."""
import re
import sys
from pathlib import Path

def update_version(version: str) -> None:
    """Update version in all relevant files."""
    # Update setup.py
    setup_path = Path('setup.py')
    setup_content = setup_path.read_text()
    setup_content = re.sub(
        r'version="[^"]*"',
        f'version="{version}"',
        setup_content
    )
    setup_path.write_text(setup_content)

    # Update __init__.py
    init_path = Path('legion/__init__.py')
    init_content = init_path.read_text()
    init_content = re.sub(
        r'__version__ = "[^"]*"',
        f'__version__ = "{version}"',
        init_content
    )
    init_path.write_text(init_content)

    # Update pyproject.toml
    pyproject_path = Path('pyproject.toml')
    if pyproject_path.exists():
        pyproject_content = pyproject_path.read_text()
        pyproject_content = re.sub(
            r'version = "[^"]*"',
            f'version = "{version}"',
            pyproject_content
        )
        pyproject_path.write_text(pyproject_content)

def main() -> int:
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python bump_version.py X.Y.Z")
        return 1

    version = sys.argv[1]
    if not re.match(r'^\d+\.\d+\.\d+$', version):
        print("Version must be in format X.Y.Z")
        return 1

    try:
        update_version(version)
        print(f"✅ Version bumped to {version}")
        print("\nNext steps:")
        print(f"1. git commit -am 'Bump version to {version}'")
        print(f"2. git tag v{version}")
        print("3. git push origin main --tags")
        return 0
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
