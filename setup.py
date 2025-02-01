from setuptools import setup, find_packages

setup(
    name="Legion-ai",
    version="0.1.0",
    description="A flexible and provider-agnostic framework for building AI agent systems.",
    url="https://github.com/LLMP-io/Legion",
    packages=find_packages(include=["legion", "legion.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "annotated-types==0.6.0",
        "anthropic==0.30.1",
        "asyncio==3.4.3",
        "bandit==1.7.8",
        "colorama==0.4.6",
        "flake8==7.0.0",
        "flake8-docstrings==1.7.0",
        "flake8-import-order==0.18.2",
        "flake8-quotes==3.4.0",
        "google-genai==0.3.0",
        "google-generativeai==0.8.2",
        "groq==0.4.2",
        "huggingface-hub==0.26.3",
        "mypy==1.9.0",
        "ollama==0.4.2",
        "openai==1.55.0",
        "psutil==5.9.8",
        "pydantic==2.10.2",
        "python-dotenv==1.0.1",
        "ruff==0.3.0",
        "safety==2.3.5",
        "types-setuptools==69.2.0.20240317",
    ],
    extras_require={
        "dev": [
            "pre-commit=4.0.1",
            "pytest=8.2.2",
            "pytest-asyncio=0.23.6",
            # "poethepoet=^0.32.1",  # Uncomment if you want to use the `poethepoet` tool
        ],
    },
    entry_points={
        "console_scripts": [
            # 'legion-cli=legion.cli:main',  # Maybe add a CLI entry point
        ],
    },
    include_package_data=True,
)
