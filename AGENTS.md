# Repository Guidelines

## Project Structure & Module Organization

Isaac GR00T N1.7 is a Python vision-language-action model stack. Core code lives in `gr00t/`: `model/` defines architectures, `data/` handles datasets, `configs/` holds configuration, `experiment/` implements training, and `policy/` and `eval/` provide inference and evaluation. `scripts/` contains deployment and conversion utilities; `examples/` contains robot and simulation integrations. Tests mirror these areas under `tests/`, with fixtures in `tests/fixtures/`. Documentation lives in `getting_started/`; media and sample datasets live in `media/` and `demo_data/`. Treat `external_dependencies/` as upstream submodules.

## Build, Test, and Development Commands

Run commands from the repository root:

- `git submodule update --init --recursive`: initialize upstream dependencies; install Git LFS for demo datasets.
- `uv sync --python 3.12 --extra dev`: install the development environment.
- `uv run pre-commit run --all-files`: run Ruff formatting, linting, and manifest alignment hooks.
- `uv run python -m pytest tests/ -m "not gpu and not edge_device and not multigpu" --timeout=300`: run CPU tests.
- `uv run python -m pytest tests/ -m gpu --timeout=300`: run GPU-marked tests on suitable hardware.
- `uv build`: build source and wheel distributions.
- `uv run python gr00t/eval/run_gr00t_server.py --model-path <checkpoint> --embodiment-tag <tag>`: start policy serving.

## Coding Style & Naming Conventions

Use Python 3.12, four-space indentation, double quotes, and a 100-character formatting target. Ruff enforces E, F, and I rules, including import sorting; configuration is in `pyproject.toml`. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Follow neighboring configuration patterns and retain existing license headers.

## Testing Guidelines

Use pytest with `test_*.py` files and `test_*` functions. Add regression tests for bug fixes and behavioral coverage for features. Mark hardware requirements with `gpu`, `edge_device`, or `multigpu`; mark resource-heavy CPU tests `serial` when needed. Run affected tests first. Coverage tooling is available through pytest-cov; no numeric coverage threshold is configured.

## Commit & Pull Request Guidelines

History uses descriptive, action-oriented subjects and occasional scoped prefixes such as `fix(validate):`; no uniform prefix is enforced. Keep commits focused. Complete `.github/pull_request_template.md`, link applicable issues with `Fixes #<number>`, describe changes and validation, update relevant docstrings, and include regression or feature tests.

## Configuration Tips

Keep Hugging Face tokens outside tracked files. Use platform-specific installation guides under `scripts/deployment/` for Jetson and Spark. When changing dependency pins, check corresponding platform manifests with `uv run python tools/check_manifest_alignment.py`.
