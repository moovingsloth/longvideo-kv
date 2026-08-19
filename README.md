# longvideo-kv

Long-video KV cache quantization experiments.

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
