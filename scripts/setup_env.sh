#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python /usr/bin/python3.12 --system-site-packages .venv
fi

# `uv pip install` does not see packages in the system/user site-packages
# made visible by --system-site-packages, so it would reinstall a fresh
# torch instead of reusing the pinned 2.11.0+cu130 wheel. The venv's own
# `python -m pip` (resolved from system dist-packages) checks the full
# sys.path and correctly treats the visible torch as already satisfied.
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [[ ! -d third-party/lmms-eval/.git ]]; then
  git clone --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git third-party/lmms-eval
fi
.venv/bin/python -m pip install -e third-party/lmms-eval

.venv/bin/python scripts/verify_environment.py
