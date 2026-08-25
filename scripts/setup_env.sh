#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

LMMS_EVAL_BASE_COMMIT="7e71cb99ddf8fe85c29e5f256dca5200cbd6211d"
LMMS_EVAL_PATCHES=(
  "${REPO_ROOT}/patches/lmms-eval-qwen-video-repro.patch"
  "${REPO_ROOT}/patches/lmms-eval-kivi-cache.patch"
  "${REPO_ROOT}/patches/lmms-eval-vidkv-cache.patch"
)

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

.venv/bin/python -m pip install -e third-party/lmms-eval

.venv/bin/python scripts/verify_environment.py
