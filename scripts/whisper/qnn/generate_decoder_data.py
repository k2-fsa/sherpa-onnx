#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import whisper

from export_onnx import AudioEncoderTensorCache, TextDecoderTensorCache, causal_mask_1d
from test_torch import compute_feat

# we need to transpose cross_kv to (1, 384, 1500) when using it as an input
# we need to transpose self_kv to (1, 384, 448) when using it as an input


def deepcopy_pair(pair):
    return [(a.clone(), b.clone()) for a, b in pair]


def to_file(tensor, filename, debug):
    if debug:
        print(filename, tensor.shape, tensor.dtype)
    tensor.numpy().tofile(filename)


@dataclass
class DecoderInput:
    tokens: torch.Tensor
    self_kv_pair: List[Tuple[torch.Tensor, torch.Tensor]]
    cross_kv_pair: List[Tuple[torch.Tensor, torch.Tensor]]
    offset: torch.Tensor
    mask: torch.Tensor

    def save_to_file(self, prefix, debug):
        ans = []
        to_file(self.tokens.to(torch.int32), f"{prefix}-tokens.raw", debug)
        ans.append(f"{prefix}-tokens.raw")

        for i, (k, v) in enumerate(self.self_kv_pair):
            to_file(k.permute(0, 2, 1), f"{prefix}_self_k_{i}.raw", debug)
            ans.append(f"{prefix}-self_k_{i}.raw")

            to_file(v.permute(0, 2, 1), f"{prefix}-self_v_{i}.raw", debug)
            ans.append(f"{prefix}-self_v_{i}.raw")

        for i, (k, v) in enumerate(self.cross_kv_pair):
            to_file(k.permute(0, 2, 1), f"{prefix}-cross_k_{i}.raw", debug)
            ans.append(f"{prefix}-self_k_{i}.raw")

            to_file(v.permute(0, 2, 1), f"{prefix}-cross_v_{i}.raw", debug)
            ans.append(f"{prefix}-self_v_{i}.raw")

        to_file(self.offset.to(torch.int32), f"{prefix}-offset.raw", debug)
        ans.append(f"{prefix}-offset.raw")

        to_file(self.mask.to(torch.int32), f"{prefix}-mask.raw", debug)
        ans.append(f"{prefix}-mask.raw")

        return ans


def process(model, tokenizer, w):
    mel = compute_feat(w)

    encoder = AudioEncoderTensorCache(model.encoder, model.decoder)
    cross_kv_pair = encoder(mel)

    # cross_kv_pair[0][0]: (1, 1500, 384)
    # cross_kv_pair[0][1]: (1, 1500, 384)

    ans = []

    decoder = TextDecoderTensorCache(model.decoder, model.dims.n_text_ctx)

    batch_size = 1
    self_kv_pair = []
    for i in range(model.dims.n_text_layer):
        k = torch.zeros(batch_size, model.dims.n_text_ctx, model.dims.n_text_state)
        v = torch.zeros(batch_size, model.dims.n_text_ctx, model.dims.n_text_state)

        self_kv_pair.append((k, v))
    # self_kv_pair[0][0]: (1, 448, 384)
    # self_kv_pair[0][1]: (1, 448, 384)

    offset = torch.zeros(1, dtype=torch.int64).to(mel.device)
    mask = causal_mask_1d(offset.item(), model.dims.n_text_ctx)

    tokens = torch.tensor([[tokenizer.sot]])

    ans.append(
        DecoderInput(
            tokens=tokens.clone(),
            self_kv_pair=deepcopy_pair(self_kv_pair),
            cross_kv_pair=deepcopy_pair(cross_kv_pair),
            offset=offset.clone(),
            mask=mask.clone(),
        )
    )

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

    ans.append(
        DecoderInput(
            tokens=tokens.clone(),
            self_kv_pair=deepcopy_pair(self_kv_pair),
            cross_kv_pair=deepcopy_pair(cross_kv_pair),
            offset=offset.clone(),
            mask=mask.clone(),
        )
    )

    for (k_cache, v_cache), (k, v) in zip(self_kv_pair, this_self_kv_pair):
        k_cache[:, offset : offset + 1] = k
        v_cache[:, offset : offset + 1] = v

    assert logits.shape == (1, tokens.shape[1], model.dims.n_vocab)

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

        ans.append(
            DecoderInput(
                tokens=tokens.clone(),
                self_kv_pair=deepcopy_pair(self_kv_pair),
                cross_kv_pair=deepcopy_pair(cross_kv_pair),
                offset=offset.clone(),
                mask=mask.clone(),
            )
        )

        for (k_cache, v_cache), (k, v) in zip(self_kv_pair, this_self_kv_pair):
            k_cache[:, offset : offset + 1] = k
            v_cache[:, offset : offset + 1] = v

        idx = logits[0, -1].argmax().item()
        steps += 1

    print(results)
    print(tokenizer.decode(results))
    return ans


@torch.no_grad()
def main():
    model = whisper.load_model("tiny.en")
    model.eval()
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages
    )

    wav_files = glob.glob("*.wav")
    features_name = []
    for w in wav_files:
        decoder_input_list = process(model, tokenizer, w)
        print(len(decoder_input_list))

        name = Path(w).stem
        files = [
            d.save_to_file(f"{name}-decoder-iter-{k:02d}", k == 0)
            for k, d in enumerate(decoder_input_list)
        ]

        features_name.extend(files)

    with open("decoder-input-list.txt", "w") as f:
        for line in features_name:
            line = " ".join(line)
            f.write(f"{line}\n")


if __name__ == "__main__":
    main()
