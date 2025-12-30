#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import whisper

from export_onnx import AudioEncoderTensorCache, TextDecoderTensorCache


def make_self_attention_mask(num_tokens: int, n: int, device=None):
    """
    Args:
        num_tokens: total sequence length
        n: number of already processed tokens
           (valid rows are [0, n-1])
    """
    i = torch.arange(num_tokens, device=device).unsqueeze(1)
    j = torch.arange(num_tokens, device=device).unsqueeze(0)

    allowed = (i < n) & (j <= i)

    mask = torch.where(
        allowed,
        torch.zeros((), device=device),
        torch.full((), float("-inf"), device=device),
    )

    return mask


#  print(make_self_attention_mask(4, 4))


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


@torch.no_grad()
def main():
    wave, sample_rate = load_audio("en.wav")
    if sample_rate != 16000:
        import librosa

        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    audio = whisper.pad_or_trim(wave)
    assert audio.shape == (16000 * 30,), audio.shape

    mel = whisper.log_mel_spectrogram(audio, n_mels=80).unsqueeze(0)
    assert mel.shape == (1, 80, 3000), mel.shape

    model = whisper.load_model("tiny.en")
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages
    )

    model.eval()

    encoder = AudioEncoderTensorCache(model.encoder, model.decoder)

    n_layer_cross_k, n_layer_cross_v = encoder(mel)
    print("n_layer_cross_k", n_layer_cross_k.shape)  # (4, 1, 1500, 384)
    print("n_layer_cross_v", n_layer_cross_v.shape)  # (4, 1, 1500, 384)

    n_audio = mel.shape[0]

    decoder = TextDecoderTensorCache(model.decoder, model.dims.n_text_ctx)

    n_layer_self_k_cache = torch.zeros(
        (
            len(model.decoder.blocks),
            n_audio,
            model.dims.n_text_ctx,
            model.dims.n_text_state,
        ),
        device=mel.device,
    )
    n_layer_self_v_cache = torch.zeros(
        (
            len(model.decoder.blocks),
            n_audio,
            model.dims.n_text_ctx,
            model.dims.n_text_state,
        ),
        device=mel.device,
    )

    offset = torch.zeros(1, dtype=torch.int64).to(mel.device)

    tokens = torch.tensor([[tokenizer.sot]])
    logits, n_layer_self_k_cache, n_layer_self_v_cache = decoder(
        tokens,
        n_layer_self_k_cache,
        n_layer_self_v_cache,
        n_layer_cross_k,
        n_layer_cross_v,
        offset,
    )
    offset += 1

    torch.save(
        {
            "self_k_offset_0": n_layer_self_k_cache,
            "self_v_offset_0": n_layer_self_v_cache,
        },
        "hyp.pt",
    )

    print(n_layer_self_k_cache.sum(), n_layer_self_v_cache.sum())

    tokens = torch.tensor([[tokenizer.no_timestamps]])
    logits, n_layer_self_k_cache, n_layer_self_v_cache = decoder(
        tokens,
        n_layer_self_k_cache,
        n_layer_self_v_cache,
        n_layer_cross_k,
        n_layer_cross_v,
        offset,
    )

    assert logits.shape == (n_audio, tokens.shape[1], model.dims.n_vocab)
    assert n_layer_self_k_cache.shape == (
        model.dims.n_text_layer,
        n_audio,
        model.dims.n_text_ctx,
        model.dims.n_text_state,
    )
    assert n_layer_self_v_cache.shape == (
        model.dims.n_text_layer,
        n_audio,
        model.dims.n_text_ctx,
        model.dims.n_text_state,
    )

    print("logits.shape", logits.shape)  # (1, 3, 51864)
    idx = logits[0, -1].argmax().item()

    steps = 0
    results = []
    while idx != tokenizer.eot and steps < 50:
        results.append(idx)
        tokens = torch.tensor([[results[-1]]])

        offset += 1

        logits, n_layer_self_k_cache, n_layer_self_v_cache = decoder(
            tokens,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            offset,
            #  mask,
        )
        idx = logits[0, -1].argmax().item()
        steps += 1

    print(results)
    print(tokenizer.decode(results))


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # To fix
    # TypeError: scaled_dot_product_attention(): argument 'is_causal' must be bool, not Tensor
    # See also https://github.com/k2-fsa/sherpa-onnx/issues/1764
    from whisper.model import disable_sdpa

    with disable_sdpa():
        main()
