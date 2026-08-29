import torch

from patches.vidkv_cache import VidkvLayer


def test_vidkv_video_mask_quantizes_only_video_tokens_and_keeps_decode_fp():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    video_mask = torch.zeros(1, 12, dtype=torch.bool)
    video_mask[:, 3:8] = True
    layer = VidkvLayer(
        nbits_key=2,
        nbits_value=2,
        q_group_size=4,
        residual_length=4,
        video_token_mask=video_mask,
    )
    key_states = torch.randn(1, 1, 12, 8, generator=generator)
    value_states = torch.randn(1, 1, 12, 8, generator=generator)

    keys, values = layer.update(key_states, value_states)

    assert keys.shape == key_states.shape
    assert values.shape == value_states.shape
    assert len(layer._quantized_keys) == 1
    assert len(layer._quantized_values) == 1
    assert layer.keys.shape[-2] == 7
    assert [(segment.quantized, segment.length) for segment in layer._segments] == [
        (False, 3),
        (True, 5),
        (False, 4),
    ]

    non_video_positions = ~video_mask[0]
    video_positions = video_mask[0]
    assert torch.equal(keys[:, :, non_video_positions, :], key_states[:, :, non_video_positions, :])
    assert torch.equal(
        values[:, :, non_video_positions, :],
        value_states[:, :, non_video_positions, :],
    )
    assert not torch.equal(keys[:, :, video_positions, :], key_states[:, :, video_positions, :])
    assert not torch.equal(
        values[:, :, video_positions, :],
        value_states[:, :, video_positions, :],
    )

    decode_key = torch.randn(1, 1, 1, 8, generator=generator)
    decode_value = torch.randn(1, 1, 1, 8, generator=generator)
    keys, values = layer.update(decode_key, decode_value)

    assert keys.shape[-2] == 13
    assert values.shape[-2] == 13
    assert len(layer._quantized_keys) == 1
    assert len(layer._quantized_values) == 1
    assert layer.keys.shape[-2] == 8
    assert (layer._segments[-1].quantized, layer._segments[-1].length) == (False, 5)
    assert torch.equal(keys[:, :, -1:, :], decode_key)
    assert torch.equal(values[:, :, -1:, :], decode_value)


def test_vidkv_prompt_layout_keeps_system_and_user_tokens_full_precision():
    """input_ids == video_token_id is True only on visual pads, not chat text.

    Without that mask VidKV quantized the whole prefill, including system/user.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2)

    # [system 4][video 6][user 5], matching a Qwen2.5-VL chat template layout.
    video_mask = torch.zeros(1, 15, dtype=torch.bool)
    video_mask[:, 4:10] = True
    layer = VidkvLayer(
        nbits_key=2,
        nbits_value=2,
        q_group_size=4,
        residual_length=4,
        video_token_mask=video_mask,
    )
    key_states = torch.randn(1, 1, 15, 8, generator=generator)
    value_states = torch.randn(1, 1, 15, 8, generator=generator)

    keys, values = layer.update(key_states, value_states)

    assert [(segment.quantized, segment.length) for segment in layer._segments] == [
        (False, 4),
        (True, 6),
        (False, 5),
    ]
    text_positions = ~video_mask[0]
    video_positions = video_mask[0]
    assert torch.equal(keys[:, :, text_positions, :], key_states[:, :, text_positions, :])
    assert torch.equal(values[:, :, text_positions, :], value_states[:, :, text_positions, :])
    assert not torch.equal(keys[:, :, video_positions, :], key_states[:, :, video_positions, :])
    assert not torch.equal(
        values[:, :, video_positions, :],
        value_states[:, :, video_positions, :],
    )


def test_vidkv_without_video_mask_keeps_prefix_quantization_behavior():
    layer = VidkvLayer(
        nbits_key=2,
        nbits_value=2,
        q_group_size=4,
        residual_length=4,
    )
    key_states = torch.randn(1, 1, 9, 8)
    value_states = torch.randn(1, 1, 9, 8)

    keys, values = layer.update(key_states, value_states)

    assert keys.shape == key_states.shape
    assert values.shape == value_states.shape
    assert len(layer._quantized_keys) == 1
    assert len(layer._quantized_values) == 1
    assert layer.keys.shape[-2] == 1
    assert layer.values.shape[-2] == 1
    assert layer._segments == []
