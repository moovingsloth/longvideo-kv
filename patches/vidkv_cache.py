from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, QuantizedLayer, get_layer_types_and_kwargs
from transformers.configuration_utils import PreTrainedConfig


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Quantization axis {axis} is out of bounds for a {ndim}D tensor.")
    return axis


def _pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        return codes.contiguous()
    values_per_byte = 8 // bits
    pad = (-codes.shape[-1]) % values_per_byte
    if pad:
        padding = torch.zeros(*codes.shape[:-1], pad, dtype=codes.dtype, device=codes.device)
        codes = torch.cat([codes, padding], dim=-1)

    grouped = codes.reshape(*codes.shape[:-1], -1, values_per_byte).to(torch.int16)
    shifts = torch.arange(values_per_byte, dtype=torch.int16, device=codes.device) * bits
    return (grouped << shifts).sum(dim=-1).to(torch.uint8).contiguous()


def _unpack_codes(packed_codes: torch.Tensor, bits: int, unpacked_size: int) -> torch.Tensor:
    if bits == 8:
        return packed_codes[..., :unpacked_size].contiguous()

    values_per_byte = 8 // bits
    shifts = torch.arange(values_per_byte, dtype=torch.int16, device=packed_codes.device) * bits
    mask = (1 << bits) - 1
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


@dataclass
class _PackedVidkvTensor:
    codes: torch.Tensor
    scale: torch.Tensor
    minimum: torch.Tensor
    original_shape: tuple[int, ...]
    packed_axis: int
    bits: float
    q_group_size: int
    mode: str

    def index_select_batch(self, indices: torch.Tensor) -> "_PackedVidkvTensor":
        return self._replace_batch(
            self.codes.index_select(0, indices.to(self.codes.device)),
            self.scale.index_select(0, indices.to(self.scale.device)),
            self.minimum.index_select(0, indices.to(self.minimum.device)),
        )

    def repeat_interleave_batch(self, repeats: int) -> "_PackedVidkvTensor":
        return self._replace_batch(
            self.codes.repeat_interleave(repeats, dim=0),
            self.scale.repeat_interleave(repeats, dim=0),
            self.minimum.repeat_interleave(repeats, dim=0),
        )

    def select_batch(self, indices: torch.Tensor) -> "_PackedVidkvTensor":
        return self._replace_batch(
            self.codes[indices.to(self.codes.device), ...],
            self.scale[indices.to(self.scale.device), ...],
            self.minimum[indices.to(self.minimum.device), ...],
        )

    def to(self, device: torch.device | str) -> "_PackedVidkvTensor":
        return _PackedVidkvTensor(
            codes=self.codes.to(device, non_blocking=True),
            scale=self.scale.to(device, non_blocking=True),
            minimum=self.minimum.to(device, non_blocking=True),
            original_shape=self.original_shape,
            packed_axis=self.packed_axis,
            bits=self.bits,
            q_group_size=self.q_group_size,
            mode=self.mode,
        )

    def _replace_batch(
        self,
        codes: torch.Tensor,
        scale: torch.Tensor,
        minimum: torch.Tensor,
    ) -> "_PackedVidkvTensor":
        return _PackedVidkvTensor(
            codes=codes,
            scale=scale,
            minimum=minimum,
            original_shape=(codes.shape[0], *self.original_shape[1:]),
            packed_axis=self.packed_axis,
            bits=self.bits,
            q_group_size=self.q_group_size,
            mode=self.mode,
        )


@dataclass
class _FftVidkvTensor:
    packed: _PackedVidkvTensor
    fft_shape: tuple[int, ...]
    output_dtype: torch.dtype

    def index_select_batch(self, indices: torch.Tensor) -> "_FftVidkvTensor":
        return _FftVidkvTensor(
            self.packed.index_select_batch(indices),
            (indices.numel(), *self.fft_shape[1:]),
            self.output_dtype,
        )

    def repeat_interleave_batch(self, repeats: int) -> "_FftVidkvTensor":
        return _FftVidkvTensor(
            self.packed.repeat_interleave_batch(repeats),
            (self.fft_shape[0] * repeats, *self.fft_shape[1:]),
            self.output_dtype,
        )

    def select_batch(self, indices: torch.Tensor) -> "_FftVidkvTensor":
        return _FftVidkvTensor(
            self.packed.select_batch(indices),
            (indices.numel(), *self.fft_shape[1:]),
            self.output_dtype,
        )

    def to(self, device: torch.device | str) -> "_FftVidkvTensor":
        return _FftVidkvTensor(self.packed.to(device), self.fft_shape, self.output_dtype)


@dataclass
class _DynamicKeyTensor:
    q_easy: _FftVidkvTensor | None
    q_difficult: _PackedVidkvTensor | None
    easy_idx: torch.Tensor
    difficult_idx: torch.Tensor
    original_shape: tuple[int, ...]
    output_dtype: torch.dtype

    def index_select_batch(self, indices: torch.Tensor) -> "_DynamicKeyTensor":
        return _DynamicKeyTensor(
            None if self.q_easy is None else self.q_easy.index_select_batch(indices),
            None if self.q_difficult is None else self.q_difficult.index_select_batch(indices),
            self.easy_idx,
            self.difficult_idx,
            (indices.numel(), *self.original_shape[1:]),
            self.output_dtype,
        )

    def repeat_interleave_batch(self, repeats: int) -> "_DynamicKeyTensor":
        return _DynamicKeyTensor(
            None if self.q_easy is None else self.q_easy.repeat_interleave_batch(repeats),
            None if self.q_difficult is None else self.q_difficult.repeat_interleave_batch(repeats),
            self.easy_idx,
            self.difficult_idx,
            (self.original_shape[0] * repeats, *self.original_shape[1:]),
            self.output_dtype,
        )

    def select_batch(self, indices: torch.Tensor) -> "_DynamicKeyTensor":
        return _DynamicKeyTensor(
            None if self.q_easy is None else self.q_easy.select_batch(indices),
            None if self.q_difficult is None else self.q_difficult.select_batch(indices),
            self.easy_idx,
            self.difficult_idx,
            (indices.numel(), *self.original_shape[1:]),
            self.output_dtype,
        )

    def to(self, device: torch.device | str) -> "_DynamicKeyTensor":
        return _DynamicKeyTensor(
            None if self.q_easy is None else self.q_easy.to(device),
            None if self.q_difficult is None else self.q_difficult.to(device),
            self.easy_idx.to(device),
            self.difficult_idx.to(device),
            self.original_shape,
            self.output_dtype,
        )


class VidkvLayer(QuantizedLayer):
    """
    VidKV-style cache layer for [batch, kv_heads, seq_len, head_dim] tensors.

    The default uses the paper/repo VLM setting: dynamic mixed key quantization
    with 1-bit FFT channels plus 2-bit anomalous channels, and 1.58-bit ternary
    value quantization. Like the official VidKV cache, retained residual tokens
    stay in full precision and grouped quantization uses q_group_size along the
    token axis for each channel (i.e. both K and V are quantized per-channel,
    which is VidKV's departure from KIVI's per-token value quantization).

    Storage is append-only: once a chunk of tokens is quantized it is never
    re-quantized, so decoding does not compound quantization error.

    Caveat on the advertised widths: `nbits_key=1.5` is honest (measured 1.52
    bits/value at head_dim=128), but `nbits_value=1.58` is stored at 2 bits/value
    because ternary codes are packed two-per-uint8. Reaching a true 1.58 would
    need 5-trits-per-byte packing (3**5 = 243 <= 256), which is not implemented.
    """

    def __init__(
        self,
        nbits_key: float = 1.5,
        nbits_value: float = 1.58,
        axis_key: int = -1,
        axis_value: int = -1,
        q_group_size: int = 32,
        residual_length: int = 128,
        ternary_threshold: float = 0.7,
    ):
        # `nbits` is the widest code width we ever pack (the 2-bit anomalous-channel and
        # ternary paths); the real per-tensor widths live in nbits_key/nbits_value.
        super().__init__(
            nbits=2,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=q_group_size,
            residual_length=residual_length,
        )
        if axis_key != -1 or axis_value != -1:
            raise ValueError("VidkvLayer currently supports axis_key=-1 and axis_value=-1 only.")
        if nbits_key not in {1, 1.5, 2, 4, 8}:
            raise ValueError("`nbits_key` must be one of {1, 1.5, 2, 4, 8}.")
        if nbits_value not in {1, 1.58, 2, 4, 8}:
            raise ValueError("`nbits_value` must be one of {1, 1.58, 2, 4, 8}.")
        if q_group_size <= 0:
            raise ValueError(f"`q_group_size` must be positive, got {q_group_size}.")
        if residual_length <= 0:
            raise ValueError(f"`residual_length` must be positive, got {residual_length}.")
        if residual_length % q_group_size != 0:
            raise ValueError(
                f"`residual_length` ({residual_length}) must be divisible by "
                f"`q_group_size` ({q_group_size})."
            )
        if not 0 < ternary_threshold < 1:
            raise ValueError(
                f"`ternary_threshold` must lie in (0, 1), got {ternary_threshold}."
            )
        self.nbits_key = nbits_key
        self.nbits_value = nbits_value
        self.ternary_threshold = ternary_threshold
        self.low_nbits = 1
        self.high_nbits = 2
        self.difficult_ratio = nbits_key - 1 if nbits_key == 1.5 else 0
        self.dynamic_key_quantize = nbits_key == 1.5
        self._quantized_keys: list[_PackedVidkvTensor | _DynamicKeyTensor | _FftVidkvTensor] = []
        self._quantized_values: list[_PackedVidkvTensor] = []

    def _quantize(
        self, tensor: torch.Tensor, axis: int, bits: float = 2
    ) -> _PackedVidkvTensor:
        if not tensor.is_floating_point():
            raise TypeError(
                f"VidKV quantization expects floating point tensors, got {tensor.dtype}."
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

        grouped = moved.reshape(*moved.shape[:-1], -1, self.q_group_size).float()
        scale = grouped.abs().mean(dim=-1, keepdim=True)
        minimum = torch.zeros_like(scale)

        if bits == 1:
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            codes = (grouped >= 0).to(torch.uint8)
            mode = "sign"
            pack_bits = 1
            scale_to_store = safe_scale
        elif bits == 1.58:
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            threshold = self.ternary_threshold * safe_scale
            codes = torch.where(grouped > threshold, 2, torch.where(grouped < -threshold, 0, 1))
            codes = codes.to(torch.uint8)
            mode = "ternary"
            pack_bits = 2
            scale_to_store = safe_scale
        else:
            bits_int = int(bits)
            minimum = grouped.amin(dim=-1, keepdim=True)
            maximum = grouped.amax(dim=-1, keepdim=True)
            scale = (maximum - minimum) / ((1 << bits_int) - 1)
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            codes = torch.round((grouped - minimum) / safe_scale)
            codes = codes.clamp_(0, (1 << bits_int) - 1).to(torch.uint8)
            mode = "affine"
            pack_bits = bits_int
            scale_to_store = scale

        return _PackedVidkvTensor(
            codes=_pack_codes(codes, pack_bits),
            scale=scale_to_store.to(dtype=tensor.dtype).contiguous(),
            minimum=minimum.to(dtype=tensor.dtype).contiguous(),
            original_shape=tuple(tensor.shape),
            packed_axis=axis,
            bits=bits,
            q_group_size=self.q_group_size,
            mode=mode,
        )

    def _dequantize(
        self,
        q_tensor: _PackedVidkvTensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        pack_bits = 2 if q_tensor.bits == 1.58 else int(q_tensor.bits)
        codes = _unpack_codes(q_tensor.codes, pack_bits, q_tensor.q_group_size)
        code_values = codes.to(dtype=q_tensor.scale.dtype)
        if q_tensor.mode == "sign":
            values = (code_values * 2 - 1) * q_tensor.scale
        elif q_tensor.mode == "ternary":
            values = (code_values - 1) * q_tensor.scale
        else:
            values = code_values * q_tensor.scale + q_tensor.minimum

        moved_shape = (*q_tensor.scale.shape[:-2], q_tensor.scale.shape[-2] * q_tensor.q_group_size)
        moved = values.reshape(moved_shape)
        moved = moved[..., : q_tensor.original_shape[q_tensor.packed_axis]]
        out = moved.movedim(-1, q_tensor.packed_axis).contiguous()
        return out if dtype is None else out.to(dtype=dtype)

    def _quantize_token_axis(self, tensor: torch.Tensor, bits: float) -> _PackedVidkvTensor:
        transformed = tensor.transpose(-2, -1).contiguous()
        return self._quantize(transformed, axis=-1, bits=bits)

    def _dequantize_token_axis(
        self,
        q_tensor: _PackedVidkvTensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self._dequantize(q_tensor, dtype=dtype).transpose(-2, -1).contiguous()

    def _quantize_fft(self, tensor: torch.Tensor, bits: float) -> _FftVidkvTensor:
        # rfft, not fft: the input is real, so the spectrum is conjugate-symmetric and the
        # second half carries no information. Storing it would double the code footprint and
        # silently turn the "1-bit" channels into 2 bits/value, breaking the nbits_key budget.
        fft_shape = tuple(tensor.shape)
        fft = torch.fft.rfft(tensor.float(), dim=-1)
        merged = torch.view_as_real(fft).reshape(*tensor.shape[:-1], -1)
        return _FftVidkvTensor(
            packed=self._quantize_token_axis(merged, bits=bits),
            fft_shape=fft_shape,
            output_dtype=tensor.dtype,
        )

    def _dequantize_fft(
        self,
        q_tensor: _FftVidkvTensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        merged = self._dequantize_token_axis(q_tensor.packed, dtype=torch.float32)
        bsz, heads, tokens, channels = q_tensor.fft_shape
        fft_real = merged.reshape(bsz, heads, tokens, -1, 2)
        fft_complex = torch.view_as_complex(fft_real.contiguous())
        out = torch.fft.irfft(fft_complex, n=channels, dim=-1)
        return out.to(dtype=dtype or q_tensor.output_dtype)

    def _quantize_key(
        self,
        tensor: torch.Tensor,
    ) -> _PackedVidkvTensor | _DynamicKeyTensor | _FftVidkvTensor:
        if self.dynamic_key_quantize:
            return self._dynamic_quantize_key(tensor)
        if self.nbits_key == 1:
            return self._quantize_fft(tensor, bits=1)
        return self._quantize_token_axis(tensor, bits=self.nbits_key)

    def _dequantize_key(
        self,
        q_tensor: _PackedVidkvTensor | _DynamicKeyTensor | _FftVidkvTensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(q_tensor, _DynamicKeyTensor):
            return self._dynamic_dequantize_key(q_tensor, dtype=dtype)
        if isinstance(q_tensor, _FftVidkvTensor):
            return self._dequantize_fft(q_tensor, dtype=dtype)
        return self._dequantize_token_axis(q_tensor, dtype=dtype)

    def _dynamic_quantize_key(self, tensor: torch.Tensor) -> _DynamicKeyTensor:
        channels = tensor.shape[-1]
        flattened = tensor.reshape(-1, channels)
        channel_range = flattened.amax(dim=0) - flattened.amin(dim=0)
        difficult_count = int(channels * self.difficult_ratio)
        if self.difficult_ratio > 0 and difficult_count == 0:
            difficult_count = min(32, channels)
        difficult_count = min(difficult_count, channels)

        if difficult_count > 0:
            difficult_idx = (
                torch.topk(channel_range, difficult_count, largest=True).indices.sort().values
            )
        else:
            difficult_idx = torch.empty(0, dtype=torch.long, device=tensor.device)
        mask = torch.ones(channels, dtype=torch.bool, device=tensor.device)
        mask[difficult_idx] = False
        easy_idx = torch.nonzero(mask, as_tuple=True)[0]

        q_easy = None
        if easy_idx.numel() > 0:
            q_easy = self._quantize_fft(torch.index_select(tensor, dim=-1, index=easy_idx), bits=1)
        q_difficult = None
        if difficult_idx.numel() > 0:
            q_difficult = self._quantize_token_axis(
                torch.index_select(tensor, dim=-1, index=difficult_idx),
                bits=2,
            )

        return _DynamicKeyTensor(
            q_easy=q_easy,
            q_difficult=q_difficult,
            easy_idx=easy_idx,
            difficult_idx=difficult_idx,
            original_shape=tuple(tensor.shape),
            output_dtype=tensor.dtype,
        )

    def _dynamic_dequantize_key(
        self,
        q_tensor: _DynamicKeyTensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        device = q_tensor.easy_idx.device
        out = torch.empty(q_tensor.original_shape, dtype=dtype, device=device)
        if q_tensor.q_easy is not None:
            easy = self._dequantize_fft(q_tensor.q_easy, dtype=dtype)
            out.index_copy_(-1, q_tensor.easy_idx, easy)
        if q_tensor.q_difficult is not None:
            difficult = self._dequantize_token_axis(q_tensor.q_difficult, dtype=dtype)
            out.index_copy_(-1, q_tensor.difficult_idx, difficult)
        return out

    def _quantize_value(self, tensor: torch.Tensor) -> _PackedVidkvTensor:
        return self._quantize_token_axis(tensor, bits=self.nbits_value)

    def _dequantize_value(self, q_tensor: _PackedVidkvTensor, dtype: torch.dtype) -> torch.Tensor:
        return self._dequantize_token_axis(q_tensor, dtype=dtype)

    def _dequantized_key_prefix(self, dtype: torch.dtype) -> torch.Tensor | None:
        if len(self._quantized_keys) == 0:
            return None
        chunks = [self._dequantize_key(chunk, dtype=dtype) for chunk in self._quantized_keys]
        return torch.cat(chunks, dim=-2) if len(chunks) > 1 else chunks[0]

    def _dequantized_value_prefix(self, dtype: torch.dtype) -> torch.Tensor | None:
        if len(self._quantized_values) == 0:
            return None
        chunks = [self._dequantize_value(chunk, dtype=dtype) for chunk in self._quantized_values]
        return torch.cat(chunks, dim=-2) if len(chunks) > 1 else chunks[0]

    def _append(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """
        Append-only flush, mirroring KiviLayer._append_keys/_append_values.

        Already-quantized chunks are never revisited: re-quantizing a dequantized chunk
        compounds its own error on every flush, which at 1-bit/1.58-bit is severe. Both K and
        V use the same rule here (unlike KIVI's asymmetric one) because VidKV quantizes both
        per-channel, so both need chunk lengths that are whole multiples of q_group_size.
        `residual_length % q_group_size == 0` is enforced in __init__, so slicing at a
        multiple of residual_length also lands on a q_group_size boundary and the padding
        path in _quantize is never taken.
        """
        combined_keys = _cat_nonempty([self.keys, key_states], dim=-2)
        combined_values = _cat_nonempty([self.values, value_states], dim=-2)
        quantized_length = (combined_keys.shape[-2] // self.residual_length) * self.residual_length
        if quantized_length > 0:
            self._quantized_keys.append(
                self._quantize_key(combined_keys[..., :quantized_length, :])
            )
            self._quantized_values.append(
                self._quantize_value(combined_values[..., :quantized_length, :])
            )
        self.keys = combined_keys[..., quantized_length:, :].contiguous()
        self.values = combined_values[..., quantized_length:, :].contiguous()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("VidkvLayer expects [batch, kv_heads, seq_len, head_dim] tensors.")
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
            self._append(key_states, value_states)
            # Prefill attention must see the quantized history too, otherwise the first
            # generated token is bit-identical to fp16 and short-answer benchmarks measure
            # nothing. After the append, prefix + residual buffer reconstructs the whole
            # prefill, so this is the same expression the decode path below uses, minus the
            # trailing new-state term.
            keys_to_return = _cat_nonempty(
                [self._dequantized_key_prefix(dtype=key_states.dtype), self.keys], dim=-2
            )
            values_to_return = _cat_nonempty(
                [self._dequantized_value_prefix(dtype=value_states.dtype), self.values], dim=-2
            )
            _check_prefill_length(keys_to_return, values_to_return, prefill_length)
            return keys_to_return, values_to_return

        key_prefix = self._dequantized_key_prefix(dtype=key_states.dtype)
        value_prefix = self._dequantized_value_prefix(dtype=value_states.dtype)
        keys_to_return = _cat_nonempty([key_prefix, self.keys, key_states], dim=-2)
        values_to_return = _cat_nonempty([value_prefix, self.values, value_states], dim=-2)

        self.cumulative_length += key_states.shape[-2]
        self._append(key_states, value_states)
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
            raise NotImplementedError("VidkvLayer.crop() is unsupported after data is quantized.")

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


class VidkvCache(Cache):
    def __init__(
        self,
        config: PreTrainedConfig,
        nbits_key: float = 1.5,
        nbits_value: float = 1.58,
        axis_key: int = -1,
        axis_value: int = -1,
        q_group_size: int = 32,
        residual_length: int = 128,
        ternary_threshold: float = 0.7,
    ):
        config = config.get_text_config(decoder=True)
        layer_types, _ = get_layer_types_and_kwargs(config)
        invalid_layer_types = set(layer_types) - {"full_attention"}
        if invalid_layer_types:
            raise ValueError(
                "`VidkvCache` is only supported for models with full attention layers. "
                f"Found invalid layer types: {sorted(invalid_layer_types)}."
            )
        layers = [
            VidkvLayer(
                nbits_key=nbits_key,
                nbits_value=nbits_value,
                axis_key=axis_key,
                axis_value=axis_value,
                q_group_size=q_group_size,
                residual_length=residual_length,
                ternary_threshold=ternary_threshold,
            )
            for _ in range(config.num_hidden_layers)
        ]
        super().__init__(layers=layers)


__all__ = ["VidkvLayer", "VidkvCache"]
