# Repository Guidelines

## Project Structure & Module Organization

This repository supports long-video KV cache quantization experiments for
`Qwen/Qwen2.5-VL-7B-Instruct`. Keep that model fixed unless `README.md` and tasks
are updated together.

- `scripts/` contains setup, environment verification, and video decode preflight tools.
- `patches/` contains cache implementations applied over `third-party/lmms-eval`:
  `fp16_cache.py`, `kivi_cache.py`, and `vidkv_cache.py`.
- `third-party/lmms-eval/` is a gitignored vendored clone recreated by setup scripts.
  Do not treat it as source owned by this repo.
- `datasets/`, `results/`, `.venv/`, `.pixi/`, and `wandb/` are local artifacts.
- Tests should live under `tests/`, matching the `pyproject.toml` pytest configuration.

## Build, Test, and Development Commands

Prefer the Pixi workflow for new work:

- `task setup` creates the Pixi environment and verifies CUDA, FlashAttention, Python,
  and video reader.
- `task setup-venv` creates the legacy uv virtualenv.
- `task smoke` runs a one-sample lmms-eval smoke test, defaulting to `mmstar`.
- `task full` runs the full evaluation, defaulting to `longvideobench_val_v`.
- `task check-video-decode` validates LongVideoBench decode behavior before a full run.
- `pixi run pytest` runs the test suite.
- `pixi run ruff check .` runs lint checks.

Override task variables inline, for example:
`task smoke FRAMES=48 OUTPUT_PATH=./results/smoke-test CUDA_VISIBLE_DEVICES=0`.

## Coding Style & Naming Conventions

Use Python 3.12.3. Follow Ruff settings from `pyproject.toml`: 100-character lines,
`py312` target, and lint rules `E`, `F`, `I`, `UP`, and `B`. Use 4-space indentation,
snake_case for functions and variables, PascalCase for classes, and lowercase file names
such as `check_video_decode.py`. Keep environment-sensitive commands explicit about
`PYTHONNOUSERSITE=1` and empty `PYTHONPATH`.

## Testing Guidelines

Use pytest. Name test files `tests/test_*.py` and test functions `test_*`. Add focused
tests for script parsing, cache behavior, and environment checks when changing those
areas. Guard GPU-dependent tests so basic collection works on unsupported machines.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, for example `Add RTX 3090 environment
support alongside GB10`. Keep commits focused and avoid checking in runtime outputs from
`results/`, `wandb/`, virtualenvs, datasets, or vendored third-party changes.

Pull requests should describe the experiment or environment impact, list commands run
(`task smoke`, `pixi run pytest`, `pixi run ruff check .`), note GPU/architecture used,
and link any relevant issue or run artifact.

## Configuration Notes

Preserve pinned versions in `pixi.toml`, `pyproject.toml`, and `requirements.txt` unless
the environment verification path is updated. The supported targets are GB10 ARM64/H100
and x86_64 RTX 3090, with the video reader pinned to `torchcodec`.
