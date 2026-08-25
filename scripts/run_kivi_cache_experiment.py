#!/usr/bin/env python3
"""Run a standalone KIVI cache storage and reconstruction experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen2_5_VLConfig
from transformers.cache_utils import get_layer_types_and_kwargs

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from patches.kivi_cache import KiviCache, KiviLayer  # noqa: E402


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_lengths(value: str) -> list[int]:
    lengths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(lengths) == 0:
        raise argparse.ArgumentTypeError("at least one length is required")
    if any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("all lengths must be positive")
    return lengths


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def quantized_bytes(layer: KiviLayer) -> int:
    total = 0
    for chunk in [*layer._quantized_keys, *layer._quantized_values]:
        total += tensor_bytes(chunk.codes)
        total += tensor_bytes(chunk.scale)
        total += tensor_bytes(chunk.minimum)
    return total


def retained_bytes(layer: KiviLayer) -> int:
    return tensor_bytes(layer.keys) + tensor_bytes(layer.values) + quantized_bytes(layer)


def key_quantized_tokens(layer: KiviLayer) -> int:
    return sum(chunk.original_shape[-1] for chunk in layer._quantized_keys)


def value_quantized_tokens(layer: KiviLayer) -> int:
    return sum(chunk.original_shape[-2] for chunk in layer._quantized_values)


def reconstruction_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
    }


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_qwen_config() -> dict[str, Any]:
    config = Qwen2_5_VLConfig()
    text_config = config.get_text_config(decoder=True)
    layer_types, _ = get_layer_types_and_kwargs(text_config)
    invalid_layer_types = sorted(set(layer_types) - {"full_attention"})
    if invalid_layer_types:
        raise RuntimeError(f"Qwen2.5-VL text config has unsupported layers: {invalid_layer_types}")

    cache = KiviCache(config)
    if len(cache.layers) != text_config.num_hidden_layers:
        raise RuntimeError(
            "KiviCache layer count mismatch: "
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


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    dtype = DTYPES[args.dtype]
    device = select_device(args.device)
    generator = make_generator(device, args.seed)

    layer = KiviLayer(
        nbits=args.nbits,
        q_group_size=args.q_group_size,
        residual_length=args.residual_length,
    )
    expected_keys: torch.Tensor | None = None
    expected_values: torch.Tensor | None = None
    records: list[dict[str, Any]] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step_index, length in enumerate(args.lengths, start=1):
        key_states = torch.randn(
            args.batch_size,
            args.kv_heads,
            length,
            args.head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        value_states = torch.randn(
            args.batch_size,
            args.kv_heads,
            length,
            args.head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        expected_keys = (
            key_states
            if expected_keys is None
            else torch.cat([expected_keys, key_states], dim=-2)
        )
        expected_values = (
            value_states
            if expected_values is None
            else torch.cat([expected_values, value_states], dim=-2)
        )

        cuda_sync(device)
        start = time.perf_counter()
        returned_keys, returned_values = layer.update(key_states, value_states)
        cuda_sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if returned_keys.shape != expected_keys.shape:
            raise RuntimeError(
                f"returned key shape mismatch: {returned_keys.shape} != {expected_keys.shape}"
            )
        if returned_values.shape != expected_values.shape:
            raise RuntimeError(
                f"returned value shape mismatch: {returned_values.shape} != {expected_values.shape}"
            )

        key_q_tokens = key_quantized_tokens(layer)
        value_q_tokens = value_quantized_tokens(layer)
        total_tokens = layer.get_seq_length()
        key_residual_tokens = layer.keys.shape[-2]
        value_residual_tokens = layer.values.shape[-2]

        if key_q_tokens + key_residual_tokens != total_tokens:
            raise RuntimeError("key quantized/residual token counts do not add up")
        if value_q_tokens + value_residual_tokens != total_tokens:
            raise RuntimeError("value quantized/residual token counts do not add up")
        if key_residual_tokens != total_tokens % args.residual_length:
            raise RuntimeError("key residual length does not match KIVI block flushing semantics")
        if value_residual_tokens != min(total_tokens, args.residual_length):
            raise RuntimeError(
                "value residual length does not keep the most recent residual_length tokens"
            )

        full_cache_bytes = tensor_bytes(expected_keys) + tensor_bytes(expected_values)
        kivi_bytes = retained_bytes(layer)
        records.append(
            {
                "step": step_index,
                "appended_tokens": length,
                "total_tokens": total_tokens,
                "elapsed_ms": elapsed_ms,
                "full_cache_bytes": full_cache_bytes,
                "kivi_retained_bytes": kivi_bytes,
                "compression_ratio": full_cache_bytes / kivi_bytes if kivi_bytes else math.inf,
                "key": {
                    "quantized_tokens": key_q_tokens,
                    "residual_tokens": key_residual_tokens,
                    "quantized_chunks": len(layer._quantized_keys),
                    "error": reconstruction_error(returned_keys, expected_keys),
                },
                "value": {
                    "quantized_tokens": value_q_tokens,
                    "residual_tokens": value_residual_tokens,
                    "quantized_chunks": len(layer._quantized_values),
                    "error": reconstruction_error(returned_values, expected_values),
                },
            }
        )

    cuda_peak_bytes = None
    if device.type == "cuda":
        cuda_peak_bytes = torch.cuda.max_memory_allocated(device)

    return {
        "config": {
            "seed": args.seed,
            "device": str(device),
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "lengths": args.lengths,
            "nbits": args.nbits,
            "q_group_size": args.q_group_size,
            "residual_length": args.residual_length,
            "qwen2_5_vl": validate_qwen_config(),
        },
        "records": records,
        "final": records[-1],
        "cuda_peak_memory_bytes": cuda_peak_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/kivi-cache-experiment.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--lengths", type=parse_lengths, default=parse_lengths("200,56,1"))
    parser.add_argument("--nbits", type=int, default=2)
    parser.add_argument("--q-group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    final = result["final"]
    print(
        "KIVI cache experiment complete: "
        f"tokens={final['total_tokens']} "
        f"compression={final['compression_ratio']:.3f}x "
        f"key_max_abs={final['key']['error']['max_abs']:.6f} "
        f"value_max_abs={final['value']['error']['max_abs']:.6f} "
        f"output={args.output_path}"
    )


if __name__ == "__main__":
    main()
