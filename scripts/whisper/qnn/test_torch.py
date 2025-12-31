#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import whisper

from export_onnx import AudioEncoderTensorCache, TextDecoderTensorCache


def causal_mask_1d(n: int, L: int, device=None, dtype=torch.float32):
    mask = torch.full((L,), float("-inf"), device=device, dtype=dtype)
    if n > 0:
        mask[:n] = 0
    return mask


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

    cross_kv_pair = encoder(mel)

    n_audio = mel.shape[0]

    decoder = TextDecoderTensorCache(model.decoder, model.dims.n_text_ctx)

    self_kv_pair = []
    for i in range(model.dims.n_text_layer):
        k = torch.zeros(n_audio, model.dims.n_text_ctx, model.dims.n_text_state)
        v = torch.zeros(n_audio, model.dims.n_text_ctx, model.dims.n_text_state)
        self_kv_pair.append((k, v))

    offset = torch.zeros(1, dtype=torch.int64).to(mel.device)

    mask = causal_mask_1d(offset.item(), model.dims.n_text_ctx)

    tokens = torch.tensor([[tokenizer.sot]])
    logits, this_self_kv_pair = decoder(
        tokens,
        self_kv_pair,
        cross_kv_pair,
        offset,
        mask,
    )
    for (k_cache, v_cache), (k, v) in zip(self_kv_pair, this_self_kv_pair):
        k_cache[:, offset : offset + 1] = k
        v_cache[:, offset : offset + 1] = v

    offset += 1

    mask = causal_mask_1d(offset.item(), model.dims.n_text_ctx)

    tokens = torch.tensor([[tokenizer.no_timestamps]])
    logits, this_self_kv_pair = decoder(
        tokens, self_kv_pair, cross_kv_pair, offset, mask
    )

    for (k_cache, v_cache), (k, v) in zip(self_kv_pair, this_self_kv_pair):
        k_cache[:, offset : offset + 1] = k
        v_cache[:, offset : offset + 1] = v

    assert logits.shape == (n_audio, tokens.shape[1], model.dims.n_vocab)

    print("logits.shape", logits.shape)  # (1, 3, 51864)
    idx = logits[0, -1].argmax().item()

    steps = 0
    results = []
    while idx != tokenizer.eot and steps < 50:
        results.append(idx)
        tokens = torch.tensor([[results[-1]]])

        offset += 1
        mask = causal_mask_1d(offset.item(), model.dims.n_text_ctx)

        logits, this_self_kv_pair = decoder(
            tokens, self_kv_pair, cross_kv_pair, offset, mask
        )

        for (k_cache, v_cache), (k, v) in zip(self_kv_pair, this_self_kv_pair):
            k_cache[:, offset : offset + 1] = k
            v_cache[:, offset : offset + 1] = v

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
