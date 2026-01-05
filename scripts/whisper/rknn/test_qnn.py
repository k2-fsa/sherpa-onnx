#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import whisper

from export_onnx import AudioEncoderTensorCache, TextDecoderTensorCache, causal_mask_1d
from test_torch import compute_feat


@torch.no_grad()
def main():
    mel = compute_feat("en.wav")

    model = whisper.load_model("tiny.en")
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages
    )

    model.eval()

    cross_kv_pair = []
    for i in range(4):
        k = features = np.fromfile(f"./cross_k_{i}.raw", dtype=np.float32).reshape(
            1, 1500, 384
        )
        v = features = np.fromfile(f"./cross_v_{i}.raw", dtype=np.float32).reshape(
            1, 1500, 384
        )

        k = torch.from_numpy(k)
        v = torch.from_numpy(v)

        cross_kv_pair.append((k, v))

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
