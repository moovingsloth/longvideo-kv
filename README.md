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

## KIVI Cache Experiment

Run the standalone KIVI cache split, reconstruction, and storage experiment:

```bash
task kivi-cache-experiment
```

The default output is `results/kivi-cache-experiment.json`.

Run lmms-eval with KIVI cache enabled:

```bash
task smoke-kivi
task full-kivi
```

These tasks pass `kv_cache=kivi` to the patched Qwen2.5-VL lmms-eval wrapper.

## VidKV Cache Experiment

[VidKV](https://arxiv.org/abs/2503.16257) compresses the KV cache below 2 bits. Keys use
mixed-precision along the channel dimension — the highest-range ("anomalous") channels get
2-bit, the rest get 1-bit in the FFT domain — and values use 1.58-bit ternary quantization.
Unlike KIVI, *both* K and V are quantized per-channel.

Run the standalone correctness, bit-budget, and error-drift experiment:

```bash
task vidkv-cache-experiment
```

The default output is `results/vidkv-cache-experiment.json`. It fails the run if the measured
key bits/value exceeds the advertised `nbits_key`, or if reconstruction error grows across
decode steps.

Run lmms-eval with VidKV cache enabled:

```bash
task smoke-vidkv
task full-vidkv
```

These pass `kv_cache=vidkv` plus `vidkv_nbits_key` / `vidkv_nbits_value` /
`vidkv_ternary_threshold` / `vidkv_q_group_size` / `vidkv_residual_length` to the wrapper.
Override any of them inline, e.g. `task full-vidkv VIDKV_NBITS_KEY=2`.

**Storage caveat:** `nbits_key=1.5` is honest — measured 1.52 bits/value at `head_dim=128`
(the extra 0.02 is the rfft's Nyquist bin). `nbits_value=1.58` is *stored* at 2 bits/value,
because ternary codes are packed two per `uint8`; a true 1.58 would need 5-trits-per-byte
packing (`3**5 = 243 <= 256`), which is not implemented. The paper's Semantic Token
Protection (STP) is also not implemented — it needs cross-modal attention scores that aren't
reachable from inside a `Cache`, and upstream only supports it for llava-onevision.
