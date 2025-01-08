# Contributing to Legion

Welcome! We're excited you're interested in contributing to Legion. This document outlines the process for contributing and helps you get started.

## Quick Links
- [Discord Server](https://discord.gg/9QnHYaAD)
- [GitHub Project Board](https://github.com/orgs/LLMP-io/projects/1)
- [Issues](https://github.com/LLMP-io/Legion/issues)

## Getting Started

1. **Fork the Repository**
   - Click the "Fork" button in the top right of the GitHub repository
   - Clone your fork locally: `git clone https://github.com/YOUR-USERNAME/Legion.git`
   - Add upstream remote: `git remote add upstream https://github.com/LLMP-io/Legion.git`

2. **Set Up Development Environment**
   ```bash
   # Run the setup script
   python3 scripts/setup_env.py

   # Activate the virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # venv\Scripts\activate
   ```

   This will:
   - Create a virtual environment
   - Install all dependencies
   - Set up pre-commit hooks

   Note: Make sure you have Python 3.8+ installed before running the setup script.

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## Development Guidelines

### Code Style
We follow PEP 8 guidelines for Python code style. Key points:
- Use 4 spaces for indentation
- Maximum line length of 79 characters for code, 72 for docstrings
- Use meaningful variable names
- Add docstrings to all functions, classes, and modules
- Use comprehensive type hints

### Type Checking
We use mypy for static type checking. Key requirements:
- All functions must have complete type hints
- All class attributes must be typed
- Use `Optional[T]` instead of `T | None`
- Use `Sequence[T]` for read-only lists
- Use `Mapping[K, V]` for read-only dicts
- Avoid `Any` unless absolutely necessary
- Run type checks before submitting PRs

### Before Submitting
1. Set up pre-commit hooks:
   ```bash
   # Install pre-commit
   pip install pre-commit

   # Install the git hooks
   pre-commit install
   ```

2. The pre-commit hooks will automatically run:
   - Non-integration tests
   - Type checking with mypy
   - Style checking with ruff
   - Security checks with bandit and safety

   You can also run these checks manually:

3. Run tests:
```bash
# Run all tests except integration tests
pytest -v -m "not integration"

# Run only integration tests
pytest -v -m integration

# Run all tests
pytest -v
```

4. Run type checking:
```bash
# Run mypy type checker
python scripts/typecheck.py

# Check specific modules
python scripts/typecheck.py legion/agents legion/blocks
```

5. Run code style checks:
```bash
# Run ruff linter
python scripts/lint.py
```

6. Run security checks:
```bash
# Run security scans
python scripts/security.py

# Check specific modules
python scripts/security.py legion/agents legion/blocks
```

7. Update documentation if needed
8. Add tests for new features

## Pull Request Process

1. **Create Pull Request**
   - Push your changes to your fork
   - Create a PR against the `main` branch
   - Fill out the PR template
   - Link any related issues

2. **PR Guidelines**
   - Keep changes focused and atomic
   - Provide clear description of changes
   - Include any necessary documentation updates
   - Add screenshots for UI changes
   - Reference any related issues

3. **Review Process**
   - Maintainers will review your PR
   - Address any requested changes
   - Once approved, maintainers will merge your PR

## Issue Guidelines

### Creating Issues
- Check existing issues to avoid duplicates
- Use issue templates when available
- Provide clear reproduction steps for bugs
- For feature requests, explain the use case

### Good First Issues
Look for issues labeled `good-first-issue` if you're new to the project. These are typically:
- Documentation improvements
- Small bug fixes
- Test additions
- Simple feature implementations

## Getting Help

- Join our Discord server for real-time discussion
- Use the #help channel for technical questions
- Tag maintainers in GitHub issues if you need clarification

## Discord Channels

- #announcements: Project updates and news
- #general: Introduce yourself to the community and ask questions
- #help: Get help with technical issues
- #development: Discuss ongoing development
- #ideas: Share and discuss feature ideas
- #pull-requests: Pull request discussions and reviews

## Recognition

We value all contributions, big and small! Contributors will be:
- Added to our Contributors list
- Recognized in release notes for significant contributions
- Badges applied to your profile within the Legion Discord Server
- Given credit in documentation when appropriate

Thank you for contributing to Legion! 🚀
