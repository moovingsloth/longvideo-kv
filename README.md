# longvideo-kv

Long-video KV cache quantization experiments.

## Supported Environments

- **GB10 (ARM64)**: H100 GPU (compute capability 12.1) on ARM64 architecture
- **RTX 3090 (x86_64)**: RTX 3090 GPU (compute capability 8.6) on x86_64 architecture

## Model

Fixed to **Qwen2.5-VL-7B-Instruct**. Do not swap models without updating this file.

## Structure

- `third-party/lmms-eval/` — vendored clone of [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), installed editable via `scripts/setup_env.sh`; gitignored, not version-controlled here
- `lmms-eval/` — Qwen2.5-VL wrapper
- `vidkv/` — quantized cache code (reference)
- `patches/` — cache implementations applied on top of `third-party/lmms-eval`
  - `fp16_cache.py` — baseline, unquantized
  - `kivi_cache.py` — KIVI quantization
  - `vidkv_cache.py` — VidKV quantization
