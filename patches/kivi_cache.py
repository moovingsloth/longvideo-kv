from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, QuantizedLayer, get_layer_types_and_kwargs
from transformers.configuration_utils import PreTrainedConfig


@dataclass
class _PackedQuantizedTensor:
    codes: torch.Tensor
    scale: torch.Tensor
    minimum: torch.Tensor
    original_shape: tuple[int, ...]
    packed_axis: int
    kind: str
    nbits: int
    q_group_size: int

    def _replace_batch(
        self,
        codes: torch.Tensor,
        scale: torch.Tensor,
        minimum: torch.Tensor,
    ) -> "_PackedQuantizedTensor":
        original_shape = (codes.shape[0], *self.original_shape[1:])
        return _PackedQuantizedTensor(
            codes=codes,
            scale=scale,
            minimum=minimum,
            original_shape=original_shape,
            packed_axis=self.packed_axis,
            kind=self.kind,
            nbits=self.nbits,
            q_group_size=self.q_group_size,
        )

    def index_select_batch(self, indices: torch.Tensor) -> "_PackedQuantizedTensor":
        if self.packed_axis == 0:
            raise NotImplementedError(
                "Batch reordering is unsupported when the quantized axis is the batch axis."
            )
        codes = self.codes.index_select(0, indices.to(self.codes.device))
        scale = self.scale.index_select(0, indices.to(self.scale.device))
        minimum = self.minimum.index_select(0, indices.to(self.minimum.device))
        return self._replace_batch(codes, scale, minimum)

    def repeat_interleave_batch(self, repeats: int) -> "_PackedQuantizedTensor":
        if self.packed_axis == 0:
            raise NotImplementedError(
                "Batch repeating is unsupported when the quantized axis is the batch axis."
            )
        return self._replace_batch(
            self.codes.repeat_interleave(repeats, dim=0),
            self.scale.repeat_interleave(repeats, dim=0),
            self.minimum.repeat_interleave(repeats, dim=0),
        )

    def select_batch(self, indices: torch.Tensor) -> "_PackedQuantizedTensor":
        if self.packed_axis == 0:
            raise NotImplementedError(
                "Batch selection is unsupported when the quantized axis is the batch axis."
            )
        return self._replace_batch(
            self.codes[indices.to(self.codes.device), ...],
            self.scale[indices.to(self.scale.device), ...],
            self.minimum[indices.to(self.minimum.device), ...],
        )

    def to(self, device: torch.device | str) -> "_PackedQuantizedTensor":
        return _PackedQuantizedTensor(
            codes=self.codes.to(device, non_blocking=True),
            scale=self.scale.to(device, non_blocking=True),
            minimum=self.minimum.to(device, non_blocking=True),
            original_shape=self.original_shape,
            packed_axis=self.packed_axis,
            kind=self.kind,
            nbits=self.nbits,
            q_group_size=self.q_group_size,
        )


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Quantization axis {axis} is out of bounds for a {ndim}D tensor.")
    return axis


def _pack_codes(codes: torch.Tensor, nbits: int) -> torch.Tensor:
    if nbits == 8:
        return codes.contiguous()

    values_per_byte = 8 // nbits
    pad = (-codes.shape[-1]) % values_per_byte
    if pad:
        padding = torch.zeros(*codes.shape[:-1], pad, dtype=codes.dtype, device=codes.device)
        codes = torch.cat([codes, padding], dim=-1)

    grouped = codes.reshape(*codes.shape[:-1], -1, values_per_byte).to(torch.int16)
    shifts = torch.arange(values_per_byte, dtype=torch.int16, device=codes.device) * nbits
    return (grouped << shifts).sum(dim=-1).to(torch.uint8).contiguous()


def _unpack_codes(packed_codes: torch.Tensor, nbits: int, unpacked_size: int) -> torch.Tensor:
    if nbits == 8:
        return packed_codes[..., :unpacked_size].contiguous()

    values_per_byte = 8 // nbits
    shifts = torch.arange(values_per_byte, dtype=torch.int16, device=packed_codes.device) * nbits
    mask = (1 << nbits) - 1
    codes = (packed_codes.to(torch.int16).unsqueeze(-1) >> shifts) & mask
    return (
        codes.reshape(*packed_codes.shape[:-1], -1)[..., :unpacked_size]
        .to(torch.uint8)
        .contiguous()
    )


def _cat_nonempty(tensors: list[torch.Tensor | None], dim: int) -> torch.Tensor:
    tensors = [tensor for tensor in tensors if tensor is not None and tensor.numel() > 0]
    if len(tensors) == 0:
        raise ValueError("Expected at least one non-empty tensor to concatenate.")
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=dim)


def _check_prefill_length(
    keys: torch.Tensor, values: torch.Tensor, expected: int
) -> None:
    """Guard the prefill reconstruction: a silent length mismatch would corrupt attention."""
    if keys.shape[-2] != expected or values.shape[-2] != expected:
        raise ValueError(
            "Prefill reconstruction lost tokens: expected "
            f"{expected}, got {keys.shape[-2]} keys and {values.shape[-2]} values."
        )


class KiviLayer(QuantizedLayer):
    """
    KIVI-style KV cache layer for tensors shaped [batch, kv_heads, seq_len, head_dim].

    Keys are quantized per channel after transposing to [B, H, D, T], with groups along
    the token axis. Values are quantized per token in [B, H, T, D], with groups along
    the channel axis. The axis_key and axis_value arguments are accepted for API
    compatibility with QuantizedCache; the KIVI update path uses the fixed axes above.
    """

    def __init__(
        self,
        nbits: int = 2,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 32,
        residual_length: int = 128,
    ):
        super().__init__(
            nbits=nbits,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=q_group_size,
            residual_length=residual_length,
        )
        if nbits not in {2, 4, 8}:
            raise ValueError(f"`nbits` must be one of {{2, 4, 8}}, got {nbits}.")
        if q_group_size <= 0:
            raise ValueError(f"`q_group_size` must be positive, got {q_group_size}.")
        if residual_length <= 0:
            raise ValueError(f"`residual_length` must be positive, got {residual_length}.")
        if residual_length % q_group_size != 0:
            raise ValueError(
                f"`residual_length` ({residual_length}) must be divisible by "
                f"`q_group_size` ({q_group_size})."
            )
        self._quantized_keys: list[_PackedQuantizedTensor] = []
        self._quantized_values: list[_PackedQuantizedTensor] = []

    def _quantize(
        self, tensor: torch.Tensor, axis: int, kind: str = "generic"
    ) -> _PackedQuantizedTensor:
        if not tensor.is_floating_point():
            raise TypeError(
                f"KIVI quantization expects a floating point tensor, got {tensor.dtype}."
            )
        if tensor.numel() == 0:
            raise ValueError("Cannot quantize an empty tensor.")

        axis = _normalize_axis(axis, tensor.ndim)
        moved = tensor.movedim(axis, -1).contiguous()
        axis_size = moved.shape[-1]
        pad = (-axis_size) % self.q_group_size
        if pad:
            padding = moved[..., -1:].expand(*moved.shape[:-1], pad)
            moved = torch.cat([moved, padding], dim=-1)

        grouped = moved.reshape(*moved.shape[:-1], -1, self.q_group_size)
        grouped_float = grouped.float()
        minimum = grouped_float.amin(dim=-1, keepdim=True)
        maximum = grouped_float.amax(dim=-1, keepdim=True)
        scale = (maximum - minimum) / ((1 << self.nbits) - 1)
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round((grouped_float - minimum) / safe_scale)
        codes = codes.clamp_(0, (1 << self.nbits) - 1).to(torch.uint8)

        return _PackedQuantizedTensor(
            codes=_pack_codes(codes, self.nbits),
            scale=scale.to(dtype=tensor.dtype).contiguous(),
            minimum=minimum.to(dtype=tensor.dtype).contiguous(),
            original_shape=tuple(tensor.shape),
            packed_axis=axis,
            kind=kind,
            nbits=self.nbits,
            q_group_size=self.q_group_size,
        )

    def _dequantize(self, q_tensor: _PackedQuantizedTensor) -> torch.Tensor:
        codes = _unpack_codes(q_tensor.codes, q_tensor.nbits, q_tensor.q_group_size)
        values = codes.to(dtype=q_tensor.scale.dtype) * q_tensor.scale + q_tensor.minimum
        moved_shape = (*q_tensor.scale.shape[:-2], q_tensor.scale.shape[-2] * q_tensor.q_group_size)
        moved = values.reshape(moved_shape)
        moved = moved[..., : q_tensor.original_shape[q_tensor.packed_axis]]
        return moved.movedim(-1, q_tensor.packed_axis).contiguous()

    def _quantize_keys(self, key_states: torch.Tensor) -> None:
        if key_states.numel() == 0:
            return
        key_states = key_states.transpose(-2, -1).contiguous()
        self._quantized_keys.append(self._quantize(key_states, axis=-1, kind="key"))

    def _quantize_values(self, value_states: torch.Tensor) -> None:
        if value_states.numel() == 0:
            return
        self._quantized_values.append(
            self._quantize(value_states.contiguous(), axis=-1, kind="value")
        )

    def _dequantized_key_prefix(self) -> torch.Tensor | None:
        if len(self._quantized_keys) == 0:
            return None
        chunks = [
            self._dequantize(chunk).transpose(-2, -1).contiguous()
            for chunk in self._quantized_keys
        ]
        return torch.cat(chunks, dim=-2)

    def _dequantized_value_prefix(self) -> torch.Tensor | None:
        if len(self._quantized_values) == 0:
            return None
        chunks = [self._dequantize(chunk) for chunk in self._quantized_values]
        return torch.cat(chunks, dim=-2)

    def _append_keys(self, key_states: torch.Tensor) -> None:
        combined = _cat_nonempty([self.keys, key_states], dim=-2)
        quantized_length = (combined.shape[-2] // self.residual_length) * self.residual_length
        self._quantize_keys(combined[..., :quantized_length, :])
        self.keys = combined[..., quantized_length:, :].contiguous()

    def _append_values(self, value_states: torch.Tensor) -> None:
        combined = _cat_nonempty([self.values, value_states], dim=-2)
        quantized_length = max(combined.shape[-2] - self.residual_length, 0)
        self._quantize_values(combined[..., :quantized_length, :])
        self.values = combined[..., quantized_length:, :].contiguous()

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError(
                "KiviLayer expects key/value tensors shaped [batch, kv_heads, seq_len, head_dim]."
            )
        if key_states.shape != value_states.shape:
            raise ValueError(
                f"Key/value shapes must match, got {key_states.shape} and {value_states.shape}."
            )

        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            prefill_length = key_states.shape[-2]
            self.cumulative_length = prefill_length
            if prefill_length == 0:
                return key_states, value_states
            self._append_keys(key_states)
            self._append_values(value_states)
            # Prefill attention must see the quantized history too, otherwise the first
            # generated token is bit-identical to fp16 and short-answer benchmarks measure
            # nothing. After the appends, prefix + residual buffer reconstructs the whole
            # prefill, so this is the same expression the decode path below uses, minus the
            # trailing new-state term.
            keys_to_return = _cat_nonempty([self._dequantized_key_prefix(), self.keys], dim=-2)
            values_to_return = _cat_nonempty(
                [self._dequantized_value_prefix(), self.values], dim=-2
            )
            _check_prefill_length(keys_to_return, values_to_return, prefill_length)
            return keys_to_return, values_to_return

        key_prefix = self._dequantized_key_prefix()
        value_prefix = self._dequantized_value_prefix()
        keys_to_return = _cat_nonempty([key_prefix, self.keys, key_states], dim=-2)
        values_to_return = _cat_nonempty([value_prefix, self.values, value_states], dim=-2)

        self.cumulative_length += key_states.shape[-2]
        self._append_keys(key_states)
        self._append_values(value_states)
        return keys_to_return, values_to_return

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_length(self) -> int:
        return -1

    def reset(self) -> None:
        self._quantized_keys = []
        self._quantized_values = []
        self.cumulative_length = 0
        if self.is_initialized:
            self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
            self.values = torch.tensor([], dtype=self.dtype, device=self.device)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() == 0:
            return
        self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
        self.values = self.values.index_select(0, beam_idx.to(self.values.device))
        self._quantized_keys = [
            chunk.index_select_batch(beam_idx) for chunk in self._quantized_keys
        ]
        self._quantized_values = [
            chunk.index_select_batch(beam_idx) for chunk in self._quantized_values
        ]

    def crop(self, tokens_to_remove: int) -> None:
        if tokens_to_remove > 0:
            current_length = self.get_seq_length()
            if tokens_to_remove >= current_length:
                return
            tokens_to_remove = current_length - tokens_to_remove
        if tokens_to_remove == 0:
            return

        if len(self._quantized_keys) > 0 or len(self._quantized_values) > 0:
            raise NotImplementedError(
                "KiviLayer.crop() is unsupported after data has been quantized."
            )

        tokens_to_remove = abs(tokens_to_remove)
        self.keys = self.keys[..., :-tokens_to_remove, :]
        self.values = self.values[..., :-tokens_to_remove, :]
        self.cumulative_length = max(self.cumulative_length - tokens_to_remove, 0)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        self.keys = self.keys.repeat_interleave(repeats, dim=0)
        self.values = self.values.repeat_interleave(repeats, dim=0)
        self._quantized_keys = [
            chunk.repeat_interleave_batch(repeats) for chunk in self._quantized_keys
        ]
        self._quantized_values = [
            chunk.repeat_interleave_batch(repeats) for chunk in self._quantized_values
        ]

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        self.keys = self.keys[indices.to(self.keys.device), ...]
        self.values = self.values[indices.to(self.values.device), ...]
        self._quantized_keys = [chunk.select_batch(indices) for chunk in self._quantized_keys]
        self._quantized_values = [chunk.select_batch(indices) for chunk in self._quantized_values]

    def offload(self) -> None:
        if not self.is_initialized:
            return
        self.keys = self.keys.to("cpu", non_blocking=True)
        self.values = self.values.to("cpu", non_blocking=True)
        self._quantized_keys = [chunk.to("cpu") for chunk in self._quantized_keys]
        self._quantized_values = [chunk.to("cpu") for chunk in self._quantized_values]

    def prefetch(self) -> None:
        if not self.is_initialized or self.keys.device == self.device:
            return
        self.keys = self.keys.to(self.device, non_blocking=True)
        self.values = self.values.to(self.device, non_blocking=True)
        self._quantized_keys = [chunk.to(self.device) for chunk in self._quantized_keys]
        self._quantized_values = [chunk.to(self.device) for chunk in self._quantized_values]


class KiviCache(Cache):
    def __init__(
        self,
        config: PreTrainedConfig,
        nbits: int = 2,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 32,
        residual_length: int = 128,
    ):
        config = config.get_text_config(decoder=True)
        layer_types, _ = get_layer_types_and_kwargs(config)
        invalid_layer_types = set(layer_types) - {"full_attention"}
        if invalid_layer_types:
            raise ValueError(
                "`KiviCache` is only supported for models with full attention layers. "
                f"Found invalid layer types: {sorted(invalid_layer_types)}."
            )

        layers = [
            KiviLayer(
                nbits=nbits,
                axis_key=axis_key,
                axis_value=axis_value,
                q_group_size=q_group_size,
                residual_length=residual_length,
            )
            for _ in range(config.num_hidden_layers)
        ]
        super().__init__(layers=layers)


__all__ = ["KiviLayer", "KiviCache"]
