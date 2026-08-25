#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

LMMS_EVAL_BASE_COMMIT="7e71cb99ddf8fe85c29e5f256dca5200cbd6211d"
LMMS_EVAL_PATCHES=(
  "${REPO_ROOT}/patches/lmms-eval-qwen-video-repro.patch"
  "${REPO_ROOT}/patches/lmms-eval-kivi-cache.patch"
)

if [[ ! -d third-party/lmms-eval/.git ]]; then
  git clone --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git third-party/lmms-eval
  git -C third-party/lmms-eval fetch --depth 1 origin "${LMMS_EVAL_BASE_COMMIT}"
  git -C third-party/lmms-eval checkout --detach "${LMMS_EVAL_BASE_COMMIT}"
fi

for lmms_eval_patch in "${LMMS_EVAL_PATCHES[@]}"; do
  if [[ -f "${lmms_eval_patch}" ]]; then
    if git -C third-party/lmms-eval apply --reverse --check "${lmms_eval_patch}" >/dev/null 2>&1; then
      echo "lmms-eval patch already applied: ${lmms_eval_patch}"
    elif git -C third-party/lmms-eval apply --check "${lmms_eval_patch}"; then
      git -C third-party/lmms-eval apply "${lmms_eval_patch}"
      echo "Applied lmms-eval patch: ${lmms_eval_patch}"
    else
      echo "Failed to apply lmms-eval patch: ${lmms_eval_patch}" >&2
      exit 1
    fi
  fi
done

PYTHONNOUSERSITE=1 PYTHONPATH= pixi install
PYTHONNOUSERSITE=1 PYTHONPATH= pixi run python scripts/verify_environment.py
PYTHONNOUSERSITE=1 PYTHONPATH= pixi run python - <<'PY'
from importlib import metadata

import flash_attn

print(f"flash-attn={metadata.version('flash-attn')}")
print(f"flash_attn.__version__={flash_attn.__version__}")
PY
