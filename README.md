# Hugging Hat

A modern Python project built using `uv`, Nix flakes, and `click`.

## Requirements

- [Nix](https://nixos.org/download.html) (with flakes enabled)
- [direnv](https://direnv.net/) (optional, but highly recommended)

## Getting Started

1. **Load the environment**:
   If you have `direnv` installed, run:
   ```bash
   direnv allow
   ```
   Otherwise, enter the Nix shell manually:
   ```bash
   nix develop
   ```

2. **Install dependencies**:
   Inside the environment, install the project dependencies using `just`:
   ```bash
   just install
   ```
   This uses `uv` under the hood to create the virtual environment and install dependencies.

3. **Run the CLI**:
   The CLI entrypoint is `hh`. You can run it via `just run`:
   ```bash
   just run hello
   ```

## Development Commands

We use `just` as our command runner. Run `just` without arguments to see all available commands.

- `just install`: Install the package and dependencies
- `just run [args...]`: Run the CLI
- `just count-lines`: Count the lines of Python code in the `src` directory
- `just build`: Build the Python wheel/sdist
