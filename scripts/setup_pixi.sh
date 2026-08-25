#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

LMMS_EVAL_BASE_COMMIT="7e71cb99ddf8fe85c29e5f256dca5200cbd6211d"
LMMS_EVAL_PATCH="${REPO_ROOT}/patches/lmms-eval-qwen-video-repro.patch"

if [[ ! -d third-party/lmms-eval/.git ]]; then
  git clone --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git third-party/lmms-eval
  git -C third-party/lmms-eval fetch --depth 1 origin "${LMMS_EVAL_BASE_COMMIT}"
  git -C third-party/lmms-eval checkout --detach "${LMMS_EVAL_BASE_COMMIT}"
fi

if [[ -f "${LMMS_EVAL_PATCH}" ]]; then
  if git -C third-party/lmms-eval apply --reverse --check "${LMMS_EVAL_PATCH}" >/dev/null 2>&1; then
    echo "lmms-eval patch already applied"
  elif git -C third-party/lmms-eval apply --check "${LMMS_EVAL_PATCH}"; then
    git -C third-party/lmms-eval apply "${LMMS_EVAL_PATCH}"
    echo "Applied lmms-eval patch: ${LMMS_EVAL_PATCH}"
  else
    echo "Failed to apply lmms-eval patch: ${LMMS_EVAL_PATCH}" >&2
    exit 1
  fi
fi

PYTHONNOUSERSITE=1 PYTHONPATH= pixi install
PYTHONNOUSERSITE=1 PYTHONPATH= pixi run python scripts/verify_environment.py
PYTHONNOUSERSITE=1 PYTHONPATH= pixi run python - <<'PY'
from importlib import metadata

import flash_attn

print(f"flash-attn={metadata.version('flash-attn')}")
print(f"flash_attn.__version__={flash_attn.__version__}")
PY
