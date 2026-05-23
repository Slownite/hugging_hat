set shell := ["bash", "-c"]

# Show available commands
default:
    @just --list

# Install dependencies
install_deps:
    uv venv --python "${UV_PYTHON:-python3}" .venv
    uv sync --extra hf --extra yaml --extra train --extra test --extra torch

# Install package
install:
    uv pip install --python .venv/bin/python -e .

# Run the CLI locally
run *ARGS:
    uv run hh {{ARGS}}

# Run the pytest suite (requires test + torch extras installed)
test *ARGS:
    .venv/bin/pytest {{ARGS}}

# Run local smoke test (requires torch extra installed)
smoke:
    PYTHONPATH=src python -m hugging_hat.testing.smoke

# Count lines of Python code in the project
count-lines:
    find src -name "*.py" -exec wc -l {} +

# Build the package
build:
    uv build

# Enter the Nix development shell (if not using direnv)
shell:
    nix develop
