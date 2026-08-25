# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Long-video KV cache quantization experiments, fixed to **Qwen2.5-VL-7B-Instruct**. Do not swap models
without updating README.md. Evaluation runs through a vendored, editable-installed clone of
[EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) at `third-party/lmms-eval/`
(gitignored, not version-controlled here — recreated by the setup scripts).

## Environments

Two supported GPU/arch combos, enforced by `scripts/verify_environment.py` (raises `SystemExit` if
the wrong one is detected):

- **GB10 (ARM64)**: H100, compute capability 12.1
- **RTX 3090 (x86_64)**: compute capability 8.6

There are two parallel environment setups — prefer pixi for new work:

- **pixi** (`pixi.toml` / `pixi.lock`): primary environment, conda-forge based, CUDA 13.0,
  Python 3.12.3 pinned.
- **legacy uv venv** (`requirements.txt` / `pyproject.toml`): `.venv/` created with
  `--system-site-packages` to reuse the system-supplied PyTorch wheel (2.11.0+cu130 on
  DGX Spark/GB10) instead of reinstalling one — see the comment block at the top of
  `requirements.txt` before touching pinned versions there.

Both setups clone `third-party/lmms-eval` if missing and end by running
`scripts/verify_environment.py`.

## Common commands

All experiment commands go through `Taskfile.yml` (the `task` CLI), which wraps
`pixi run accelerate launch -m lmms_eval`. Key tasks:

```bash
task setup              # pixi env + CUDA/FlashAttention2 verification
task setup-venv         # legacy uv venv equivalent
task smoke              # 1-sample lmms-eval smoke test (default task: mmstar)
task full               # full lmms-eval run (default task: longvideobench_val_v)
task check-video-decode # preflight video decoding with the pinned backend/reader
```

Override task variables inline, e.g.:

```bash
task full TASKS=longvideobench_val_v FRAMES=48 MIN_PIXELS=200704 MAX_PIXELS=401408 GPUS=1
task smoke WANDB=my-run-name
```

Notable variables (see `Taskfile.yml` for the full list/defaults): `FRAMES`, `MIN_PIXELS`,
`MAX_PIXELS`, `FPS`, `VIDEO_READER` (defaults to `torchcodec`), `STRICT_VIDEO_READER`,
`ATTN_IMPLEMENTATION` (defaults to `flash_attention_2`), `GPUS`, `OUTPUT_PATH` (defaults to
`./results`), `CUDA_VISIBLE_DEVICES`.

Run directly with pixi when not using a Taskfile wrapper: `pixi run <cmd>`. Always run with
`PYTHONNOUSERSITE=1 PYTHONPATH=` (as the tasks/scripts do) to avoid leaking user/system site-packages
into the pinned environment.

Lint (uv/legacy env, per `pyproject.toml`): `ruff check .` (line-length 100, target py312,
rules `E,F,I,UP,B`). Tests: `pytest` (`testpaths = ["tests"]`; no `tests/` directory exists yet).

## Architecture

- `patches/` — KV cache implementations applied on top of `third-party/lmms-eval`: `fp16_cache.py`
  (baseline, unquantized — still an empty placeholder), `kivi_cache.py` (KIVI quantization),
  `vidkv_cache.py` (VidKV quantization). KIVI and VidKV are implemented as `Cache`/`QuantizedLayer`
  subclasses selected via the `kv_cache=` model arg. Both store quantized chunks **append-only**:
  a chunk is never re-quantized once written, because re-quantizing dequantized history compounds
  error on every flush. `patches/` is importable only because tasks run with cwd = repo root
  (`accelerate launch -m` puts cwd on `sys.path`); `PYTHONPATH` is deliberately blanked.
- `patches/lmms-eval-*.patch` — edits to the vendored harness, applied in array order by both
  setup scripts. Regenerate a patch by diffing against the *previous* patch's output, not
  against HEAD, since they stack on the same files.
- `third-party/lmms-eval/` — vendored upstream eval harness; not modified directly, gitignored.
  Editable-installed via pip (`setup_env.sh`) or as a pypi-dependency path in `pixi.toml`.
- `scripts/verify_environment.py` — single source of truth for "is this the right GPU/arch/driver
  stack"; both setup scripts run it as their last step.
- `scripts/check_video_decode.py` — standalone preflight script for validating LongVideoBench video
  decoding against the same fixed video-reader backend (`FORCE_QWENVL_VIDEO_READER`, default
  `torchcodec`) used during actual evaluation, before committing to a full run.
- `results/` — gitignored output of `task smoke`/`task full` runs (JSON results, decode-preflight
  JSONL logs), organized by run type / model name / timestamp.
- `wandb/` — gitignored local W&B run logs from tasks invoked with `wandb=`/`WANDB=`.

## Key constraints to preserve

- Python interpreter is pinned to exactly 3.12.3 in both environments; a different minor/patch
  won't share the system site-packages the uv venv depends on.
- Video decoding backend is pinned to `torchcodec` (`FORCE_QWENVL_VIDEO_READER`,
  `STRICT_VIDEO_READER=1`) so evaluation and preflight checks use identical decode behavior.
- `decord2` (not `decord`) is used on aarch64 because upstream `decord` has no linux-aarch64
  wheels; it installs as the same `decord` module so `lmms-eval`'s `import decord` works
  unmodified.
