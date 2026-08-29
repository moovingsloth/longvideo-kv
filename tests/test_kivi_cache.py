import torch

from patches.kivi_cache import KiviLayer


def test_kivi_prefill_returns_quantized_history_not_original_fp16():
    """Prefill attention must see dequantized K/V, not the original tensors.

    HuggingFace QuantizedLayer.update() stores quantized codes then returns the
    incoming key_states/value_states unchanged. That made VideoMME match the
    FP16 baseline: the first generated token is produced from prefill attention.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    layer = KiviLayer(nbits=2, q_group_size=4, residual_length=8)
    key_states = torch.randn(1, 1, 20, 8, generator=generator)
    value_states = torch.randn(1, 1, 20, 8, generator=generator)

    keys, values = layer.update(key_states, value_states)

    assert keys.shape == key_states.shape
    assert values.shape == value_states.shape
    assert len(layer._quantized_keys) == 1
    assert len(layer._quantized_values) == 1
    assert layer.keys.shape[-2] == 4
    assert layer.values.shape[-2] == 8
    assert not torch.equal(keys, key_states)
    assert not torch.equal(values, value_states)

    # Residual windows stay full precision; only the flushed prefix is quantized.
    assert torch.equal(keys[:, :, -4:, :], key_states[:, :, -4:, :])
    assert torch.equal(values[:, :, -8:, :], value_states[:, :, -8:, :])
    assert not torch.equal(keys[:, :, :16, :], key_states[:, :, :16, :])
    assert not torch.equal(values[:, :, :12, :], value_states[:, :, :12, :])


def test_kivi_decode_reuses_quantized_prefix_without_requantizing_it():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)

    layer = KiviLayer(nbits=2, q_group_size=4, residual_length=8)
    prefill_keys = torch.randn(1, 1, 16, 8, generator=generator)
    prefill_values = torch.randn(1, 1, 16, 8, generator=generator)
    keys, values = layer.update(prefill_keys, prefill_values)
    quantized_key_chunks = len(layer._quantized_keys)

    decode_key = torch.randn(1, 1, 1, 8, generator=generator)
    decode_value = torch.randn(1, 1, 1, 8, generator=generator)
    keys, values = layer.update(decode_key, decode_value)

    assert keys.shape[-2] == 17
    assert values.shape[-2] == 17
    # Keys flush in residual_length blocks, so one decode token stays residual.
    assert len(layer._quantized_keys) == quantized_key_chunks
    # Values keep only the last residual_length tokens in FP16, so the oldest
    # residual value is flushed into a new quantized chunk on this step.
    assert len(layer._quantized_values) == 2
    assert layer.values.shape[-2] == 8
    assert torch.equal(keys[:, :, -1:, :], decode_key)
    assert torch.equal(values[:, :, -1:, :], decode_value)
    assert not torch.equal(keys[:, :, :8, :], prefill_keys[:, :, :8, :])
    assert not torch.equal(values[:, :, :8, :], prefill_values[:, :, :8, :])
