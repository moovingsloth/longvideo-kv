#!/usr/bin/env python3
"""Run a standalone VidKV cache correctness, bit-budget, and error-drift experiment.

Three regression guards, matching the three defects this implementation was fixed for:

1. round-trip   -- key/value quantize->dequantize reconstruction error.
2. bit budget   -- measured code bits/value must match the advertised ``nbits_key``.
                   A full complex FFT on real input silently doubles this to 2.0.
3. error drift  -- error must stay flat across decode steps. Re-quantizing already
                   dequantized history compounds error monotonically instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen2_5_VLConfig
from transformers.cache_utils import get_layer_types_and_kwargs

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from patches.vidkv_cache import (  # noqa: E402
    VidkvCache,
    VidkvLayer,
    _DynamicKeyTensor,
    _FftVidkvTensor,
)

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Ternary codes pack two-per-uint8, so a "1.58-bit" value cache costs 2.0 bits/value.
# See the caveat in VidkvLayer's docstring.
VALUE_STORED_BITS = {1: 1.0, 1.58: 2.0, 2: 2.0, 4: 4.0, 8: 8.0}


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def code_bits(q_tensor: Any) -> int:
    """Total bits held in packed code words, excluding per-group scale/minimum overhead."""
    if isinstance(q_tensor, _DynamicKeyTensor):
        total = 0
        if q_tensor.q_easy is not None:
            total += code_bits(q_tensor.q_easy)
        if q_tensor.q_difficult is not None:
            total += code_bits(q_tensor.q_difficult)
        return total
    if isinstance(q_tensor, _FftVidkvTensor):
        return code_bits(q_tensor.packed)
    return q_tensor.codes.numel() * 8


def metadata_bits(q_tensor: Any) -> int:
    """Bits held in per-group scale/minimum tensors."""
    if isinstance(q_tensor, _DynamicKeyTensor):
        total = 0
        if q_tensor.q_easy is not None:
            total += metadata_bits(q_tensor.q_easy)
        if q_tensor.q_difficult is not None:
            total += metadata_bits(q_tensor.q_difficult)
        return total
    if isinstance(q_tensor, _FftVidkvTensor):
        return metadata_bits(q_tensor.packed)
    packed = q_tensor
    return (packed.scale.numel() + packed.minimum.numel()) * packed.scale.element_size() * 8


def rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    )


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "rel_error": rel_error(actual, expected),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
    }


def validate_qwen_config(args: argparse.Namespace) -> dict[str, Any]:
    config = Qwen2_5_VLConfig()
    text_config = config.get_text_config(decoder=True)
    layer_types, _ = get_layer_types_and_kwargs(text_config)
    invalid_layer_types = sorted(set(layer_types) - {"full_attention"})
    if invalid_layer_types:
        raise RuntimeError(f"Qwen2.5-VL text config has unsupported layers: {invalid_layer_types}")

    cache = VidkvCache(
        config,
        nbits_key=args.nbits_key,
        nbits_value=args.nbits_value,
        q_group_size=args.q_group_size,
        residual_length=args.residual_length,
        ternary_threshold=args.ternary_threshold,
    )
    if len(cache.layers) != text_config.num_hidden_layers:
        raise RuntimeError(
            "VidkvCache layer count mismatch: "
            f"{len(cache.layers)} != {text_config.num_hidden_layers}"
        )

    return {
        "num_hidden_layers": text_config.num_hidden_layers,
        "num_attention_heads": text_config.num_attention_heads,
        "num_key_value_heads": text_config.num_key_value_heads,
        "hidden_size": text_config.hidden_size,
        "head_dim": text_config.hidden_size // text_config.num_attention_heads,
        "layer_types": sorted(set(layer_types)),
    }


def make_layer(args: argparse.Namespace) -> VidkvLayer:
    return VidkvLayer(
        nbits_key=args.nbits_key,
        nbits_value=args.nbits_value,
        q_group_size=args.q_group_size,
        residual_length=args.residual_length,
        ternary_threshold=args.ternary_threshold,
    )


def check_round_trip(args: argparse.Namespace, device, dtype, generator) -> dict[str, Any]:
    """Guard 1 + 2: reconstruction error and measured bits/value."""
    layer = make_layer(args)
    shape = (args.batch_size, args.kv_heads, args.tokens, args.head_dim)
    tensor = torch.randn(shape, generator=generator, device=device, dtype=torch.float32).to(dtype)
    elements = tensor.numel()

    q_key = layer._quantize_key(tensor)
    q_value = layer._quantize_value(tensor)
    key_bits = code_bits(q_key) / elements
    value_bits = code_bits(q_value) / elements

    result = {
        "shape": list(shape),
        "key": {
            "advertised_bits": args.nbits_key,
            "measured_bits_per_value": key_bits,
            "metadata_bits_per_value": metadata_bits(q_key) / elements,
            **error_stats(layer._dequantize_key(q_key, dtype=dtype), tensor),
        },
        "value": {
            "advertised_bits": args.nbits_value,
            "stored_bits": VALUE_STORED_BITS[args.nbits_value],
            "measured_bits_per_value": value_bits,
            "metadata_bits_per_value": metadata_bits(q_value) / elements,
            **error_stats(layer._dequantize_value(q_value, dtype=dtype), tensor),
        },
    }

    # The key budget is the headline claim: 1.5-bit must actually be ~1.5 bits.
    # Tolerance covers the rfft's +1 complex bin (2*(D//2+1)/D, i.e. +2/D per easy channel).
    key_tolerance = 4.0 / args.head_dim
    result["key"]["within_budget"] = key_bits <= args.nbits_key + key_tolerance
    result["value"]["within_budget"] = value_bits <= VALUE_STORED_BITS[args.nbits_value] + 1e-6
    return result


def check_error_drift(args: argparse.Namespace, device, dtype, generator) -> dict[str, Any]:
    """Guard 3: error must not compound as tokens are decoded one at a time."""
    layer = make_layer(args)
    prefill = torch.randn(
        (args.batch_size, args.kv_heads, args.prefill_tokens, args.head_dim),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(dtype)
    layer.update(prefill, prefill.clone())
    truth = [prefill]

    checkpoints = {1, args.decode_steps // 4, args.decode_steps // 2, args.decode_steps}
    samples: list[dict[str, Any]] = []
    for step in range(1, args.decode_steps + 1):
        token = torch.randn(
            (args.batch_size, args.kv_heads, 1, args.head_dim),
            generator=generator,
            device=device,
            dtype=torch.float32,
        ).to(dtype)
        truth.append(token)
        keys, values = layer.update(token, token.clone())
        if step in checkpoints:
            expected = torch.cat(truth, dim=-2)
            if keys.shape != expected.shape:
                raise RuntimeError(f"cache shape {tuple(keys.shape)} != {tuple(expected.shape)}")
            samples.append(
                {
                    "step": step,
                    "seq_length": layer.get_seq_length(),
                    "quantized_chunks": len(layer._quantized_keys),
                    "residual_tokens": layer.keys.shape[-2],
                    "key_rel_error": rel_error(keys, expected),
                    "value_rel_error": rel_error(values, expected),
                }
            )

    first, last = samples[0], samples[-1]
    # Destructive re-quantization grew this ~22% over 300 steps; append-only holds it flat.
    drift = last["key_rel_error"] / first["key_rel_error"] if first["key_rel_error"] else 1.0
    return {
        "prefill_tokens": args.prefill_tokens,
        "decode_steps": args.decode_steps,
        "samples": samples,
        "key_error_ratio_last_over_first": drift,
        "residual_bounded": all(s["residual_tokens"] < args.residual_length for s in samples),
        "no_drift": drift <= args.max_drift,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    dtype = DTYPES[args.dtype]
    device = select_device(args.device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(args.seed)

    config = validate_qwen_config(args)
    round_trip = check_round_trip(args, device, dtype, generator)
    drift = check_error_drift(args, device, dtype, generator)

    passed = (
        round_trip["key"]["within_budget"]
        and round_trip["value"]["within_budget"]
        and drift["no_drift"]
        and drift["residual_bounded"]
    )
    return {
        "settings": {
            "seed": args.seed,
            "device": str(device),
            "dtype": args.dtype,
            "nbits_key": args.nbits_key,
            "nbits_value": args.nbits_value,
            "ternary_threshold": args.ternary_threshold,
            "q_group_size": args.q_group_size,
            "residual_length": args.residual_length,
        },
        "qwen_config": config,
        "round_trip": round_trip,
        "error_drift": drift,
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=300)
    parser.add_argument("--nbits-key", type=float, default=1.5)
    parser.add_argument("--nbits-value", type=float, default=1.58)
    parser.add_argument("--ternary-threshold", type=float, default=0.7)
    parser.add_argument("--q-group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument(
        "--max-drift",
        type=float,
        default=1.05,
        help="max tolerated ratio of last-step to first-step key error",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_experiment(args)

    rt, drift = report["round_trip"], report["error_drift"]
    print(
        f"key   : {rt['key']['measured_bits_per_value']:.3f} bits/value "
        f"(advertised {args.nbits_key})  rel_error {rt['key']['rel_error']:.4f}  "
        f"{'OK' if rt['key']['within_budget'] else 'OVER BUDGET'}"
    )
    print(
        f"value : {rt['value']['measured_bits_per_value']:.3f} bits/value "
        f"(advertised {args.nbits_value}, stored {rt['value']['stored_bits']})  "
        f"rel_error {rt['value']['rel_error']:.4f}  "
        f"{'OK' if rt['value']['within_budget'] else 'OVER BUDGET'}"
    )
    for sample in drift["samples"]:
        print(
            f"  step {sample['step']:4d}: key {sample['key_rel_error']:.4f}  "
            f"value {sample['value_rel_error']:.4f}  "
            f"chunks {sample['quantized_chunks']}  residual {sample['residual_tokens']}"
        )
    print(
        f"drift : last/first = {drift['key_error_ratio_last_over_first']:.3f} "
        f"(max {args.max_drift})  {'OK' if drift['no_drift'] else 'ERROR COMPOUNDING'}"
    )

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output_path}")

    if not report["passed"]:
        print("FAILED", file=sys.stderr)
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
