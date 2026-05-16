set shell := ["bash", "-c"]

# Show available commands
default:
    @just --list

# Install dependencies
install:
    uv sync

# Run the CLI locally
run *ARGS:
    uv run hh {{ARGS}}

# Count lines of Python code in the project
count-lines:
    find src -name "*.py" -exec wc -l {} +

# Build the package
build:
    uv build

# Enter the Nix development shell (if not using direnv)
shell:
    nix develop
