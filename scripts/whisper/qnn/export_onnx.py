#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
# flake8: noqa

"""
Note: Code in this file is modified from
https://github.com/TadaoYamaoka/whisper/blob/main/to_onnx.py

Thanks to https://github.com/TadaoYamaoka
for making the onnx export script public.

Note that we have removed the 30 seconds constraint from whisper. You can
use any T <= 30.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import onnx
import torch
import torch.nn.functional as F
import whisper
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor, nn
from whisper.model import (
    AudioEncoder,
    MultiHeadAttention,
    ResidualAttentionBlock,
    TextDecoder,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        # fmt: off
        choices=[
            "tiny", "tiny.en", "base", "base.en",
            "small", "small.en", "medium", "medium.en",
            "large-v1", "large-v2",
            "large", "large-v3", "turbo", # these three have feature dim 128
            "distil-medium.en", "distil-small.en", "distil-large-v2",
            "distil-large-v3",
            "distil-large-v3.5",
            # for fine-tuned models from icefall
            "medium-aishell",
            ],
        # fmt: on
    )
    return parser.parse_args()


def causal_mask_1d(n: int, L: int, device=None, dtype=torch.int32):
    """
    Returns a 1-D int mask of shape (L,) with:
      0 -> allowed
      1 -> masked (will be converted to -inf later)
    """
    mask = torch.ones((L,), device=device, dtype=dtype)
    if n > 0:
        mask[:n] = 0
    return mask


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    if "large" in filename or "turbo" in filename:
        external_filename = filename.split(".onnx")[0]
        onnx.save(
            model,
            filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_filename + ".weights",
        )
    else:
        onnx.save(model, filename)


def modified_self_qkv_attention(
    self,
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    k1: Tensor,
    v1: Tensor,
    mask: Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    assert mask is not None

    n_batch, n_ctx, n_state = q.shape

    scale = (n_state // self.n_head) ** -0.25
    q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    k_cache = k_cache.view(*k_cache.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    v_cache = v_cache.view(*v_cache.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

    k1 = k1.view(*k1.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    v1 = v1.view(*v1.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

    qk = (q * scale) @ (k_cache * scale).transpose(-1, -2)  # (1, 6, 1, 448)

    qk1 = (q * scale) @ (k1 * scale).transpose(-1, -2)  # (1, 6, 1, 1)

    #  qk = qk + mask
    qk.masked_fill_(mask.to(torch.bool), float("-inf"))

    qk = qk.float()
    qk1 = qk1.float()

    qk_total = torch.cat([qk, qk1], dim=-1)

    w_total = F.softmax(qk_total, dim=-1).to(q.dtype)
    w = w_total[:, :, :, :-1]
    w1 = w_total[:, :, :, -1:]

    out = (w @ v_cache).permute(0, 2, 1, 3).flatten(start_dim=2)
    out1 = (w1 @ v1).permute(0, 2, 1, 3).flatten(start_dim=2)
    out = out + out1

    qk = qk.detach()

    return out, qk


MultiHeadAttention.qkv_attention_self = modified_self_qkv_attention


def modified_audio_encoder_forward(self: AudioEncoder, x: torch.Tensor):
    """
    x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
        the mel spectrogram of the audio
    """
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)

    if False:
        # This branch contains the original code
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
    else:
        #  print(x.shape, self.positional_embedding.shape)
        # This branch contains the actual changes
        assert (
            x.shape[2] == self.positional_embedding.shape[1]
        ), f"incorrect audio shape: {x.shape}, {self.positional_embedding.shape}"
        assert (
            x.shape[1] == self.positional_embedding.shape[0]
        ), f"incorrect audio shape: {x.shape}, {self.positional_embedding.shape}"
        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

    for block in self.blocks:
        x = block(x)

    x = self.ln_post(x)
    return x


AudioEncoder.forward = modified_audio_encoder_forward


class AudioEncoderTensorCache(nn.Module):
    def __init__(self, inAudioEncoder: AudioEncoder, inTextDecoder: TextDecoder):
        super().__init__()
        self.audioEncoder = inAudioEncoder
        self.textDecoder = inTextDecoder

    def forward(self, x: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """
        Args:
          x: (1, 80, 3000)
          cross_kv_pair:
            - the i-th entry contains kv cache for the i-th layer
        """
        audio_features = self.audioEncoder(x)

        n_layer_cross_k_list = []
        n_layer_cross_v_list = []

        cross_kv_pair = []
        for block in self.textDecoder.blocks:
            k = block.cross_attn.key(audio_features)  # (batch_size, 1500, 384)
            v = block.cross_attn.value(audio_features)  # (batch_size, 1500, 384)

            cross_kv_pair.append((k, v))

        return cross_kv_pair


class MultiHeadAttentionCross(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(
        self,
        x: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.multiHeadAttention.query(x)
        wv, qk = self.multiHeadAttention.qkv_attention(q, k, v, mask)
        return self.multiHeadAttention.out(wv)


class MultiHeadAttentionSelf(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(
        self,
        x: Tensor,  # (1, 1      , 384)
        k_cache: Tensor,  # (1, 448, 384)
        v_cache: Tensor,  # (1, 448, 384)
        mask: Tensor,  # (448,)
    ):
        q = self.multiHeadAttention.query(x)  # (1, 1, 384)
        k = self.multiHeadAttention.key(x)  # (1, 1, 384)
        v = self.multiHeadAttention.value(x)  # (1, 1, 384)

        #  k_cache[:, offset : offset + 1, :] = k  # (b, n_ctx_cache + n_ctx, n_state)
        #  v_cache[:, offset : offset + 1, :] = v  # (b, n_ctx_cache + n_ctx, n_state)

        wv, qk = self.multiHeadAttention.qkv_attention_self(
            q,
            k_cache=k_cache,
            v_cache=v_cache,
            k1=k,
            v1=v,
            mask=mask,
        )

        return self.multiHeadAttention.out(wv), k, v


class ResidualAttentionBlockTensorCache(nn.Module):
    def __init__(self, inResidualAttentionBlock: ResidualAttentionBlock):
        super().__init__()
        self.originalBlock = inResidualAttentionBlock
        self.attn = MultiHeadAttentionSelf(inResidualAttentionBlock.attn)
        self.cross_attn = (
            MultiHeadAttentionCross(inResidualAttentionBlock.cross_attn)
            if inResidualAttentionBlock.cross_attn
            else None
        )

    def forward(
        self,
        x: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        offset: Tensor,
        mask: Tensor,
    ):
        self_attn_x, self_k, self_v = self.attn(
            self.originalBlock.attn_ln(x),
            self_k_cache,
            self_v_cache,
            mask=mask,
        )
        x = x + self_attn_x

        if self.cross_attn:
            x = x + self.cross_attn(
                self.originalBlock.cross_attn_ln(x), cross_k, cross_v
            )

        x = x + self.originalBlock.mlp(self.originalBlock.mlp_ln(x))
        return x, self_k, self_v


class TextDecoderTensorCache(nn.Module):
    def __init__(self, inTextDecoder: TextDecoder, in_n_ctx: int):
        super().__init__()
        self.textDecoder = inTextDecoder
        self.n_ctx = in_n_ctx

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlockTensorCache(orginal_block))

    def forward(
        self,
        tokens: Tensor,
        self_kv_pair: List[Tuple[Tensor, Tensor]],
        cross_kv_pair: List[Tuple[Tensor, Tensor]],
        offset: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        tokens: (batch_size, 1)
        self_kv_pair:
            - [i][0]: layer_i_self_k_cache, (batch_size, 448, dim)
            - [i][1]: layer_i_self_v_cache, (batch_size, 448, dim)
        Returns:
          - logits
          - this_self_kv_pair
        """
        assert tokens.shape == (1, 1), tokens.shape
        x = (
            self.textDecoder.token_embedding(tokens)
            + self.textDecoder.positional_embedding[offset[0] : offset[0] + 1]
        )

        i = 0
        this_self_kv_pair = []
        for block in self.blocks:
            self_k_cache = self_kv_pair[i][0]
            self_v_cache = self_kv_pair[i][1]

            x, self_k, self_v = block(
                x,
                #  self_k_cache=self_k_cache[:, : offset + 1],
                #  self_v_cache=self_v_cache[:, : offset + 1],
                self_k_cache=self_k_cache,
                self_v_cache=self_v_cache,
                cross_k=cross_kv_pair[i][0],
                cross_v=cross_kv_pair[i][1],
                offset=offset,
                #  mask=self.textDecoder.mask,
                mask=mask,
            )
            #  self_k_cache[:, : offset + 1] = updated_self_k_cache
            #  self_v_cache[:, : offset + 1] = updated_self_v_cache
            #  updated_self_kv_pair.append((self_k_cache, self_v_cache))
            this_self_kv_pair.append((self_k, self_v))

            i += 1

        x = self.textDecoder.ln(x)

        if False:
            # x.shape (1, 3, 384)
            # weight.shape (51684, 384)

            logits = (
                x
                @ torch.transpose(
                    self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1
                )
            ).float()
        else:
            logits = (
                torch.matmul(
                    self.textDecoder.token_embedding.weight.to(x.dtype),
                    x.permute(0, 2, 1),
                )
                .permute(0, 2, 1)
                .float()
            )

        return logits, this_self_kv_pair


# ref: https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-pt-to-ggml.py#L232
def convert_tokens(name, model):
    whisper_dir = Path(whisper.__file__).parent
    multilingual = model.is_multilingual
    tokenizer = (
        whisper_dir
        / "assets"
        / (multilingual and "multilingual.tiktoken" or "gpt2.tiktoken")
    )
    if not tokenizer.is_file():
        raise ValueError(f"Cannot find {tokenizer}")

    #  import base64

    with open(tokenizer, "r") as f:
        contents = f.read()
        #  tokens = {
        #      base64.b64decode(token): int(rank)
        #      for token, rank in (line.split() for line in contents.splitlines() if line)
        #  }
        tokens = {
            token: int(rank)
            for token, rank in (line.split() for line in contents.splitlines() if line)
        }

    with open(f"{name}-tokens.txt", "w") as f:
        for t, i in tokens.items():
            f.write(f"{t} {i}\n")


@torch.no_grad()
def main():
    args = get_args()
    name = args.model
    print(args)
    print(name)

    opset_version = 17

    if name == "distil-medium.en":
        filename = "./distil-medium-en-original-model.bin"
        if not Path(filename).is_file():
            raise ValueError(
                """
                Please go to https://huggingface.co/distil-whisper/distil-medium.en
                to download original-model.bin
                You can use the following command to do that:

                wget -O distil-medium-en-original-model.bin https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/original-model.bin
            """
            )
        model = whisper.load_model(filename)
    elif name == "distil-large-v2":
        filename = "./distil-large-v2-original-model.bin"
        if not Path(filename).is_file():
            raise ValueError(
                """
                Please go to https://huggingface.co/distil-whisper/distil-large-v2
                to download original-model.bin
                You can use the following command to do that:

                wget -O distil-large-v2-original-model.bin https://huggingface.co/distil-whisper/distil-large-v2/resolve/main/original-model.bin
            """
            )
        model = whisper.load_model(filename)
    elif name == "distil-large-v3":
        filename = "./distil-large-v3-original-model.bin"
        if not Path(filename).is_file():
            raise ValueError(
                """
                Please go to https://huggingface.co/distil-whisper/distil-large-v3-openai
                to download model.bin
                You can use the following command to do that:

                wget -O distil-large-v3-original-model.bin https://huggingface.co/distil-whisper/distil-large-v3-openai/resolve/main/model.bin
            """
            )
        model = whisper.load_model(filename)
    elif name == "distil-large-v3.5":
        filename = "./distil-large-v3.5-original-model.bin"
        if not Path(filename).is_file():
            raise ValueError(
                """
                Please go to https://huggingface.co/distil-whisper/distil-large-v3.5-openai/
                to download model.bin
                You can use the following command to do that:

                wget -O distil-large-v3.5-original-model.bin https://huggingface.co/distil-whisper/distil-large-v3.5-openai/resolve/main/model.bin
            """
            )
        model = whisper.load_model(filename)
    elif name == "distil-small.en":
        filename = "./distil-small-en-original-model.bin"
        if not Path(filename).is_file():
            raise ValueError(
                """
                Please go to https://huggingface.co/distil-whisper/distil-small.en
                to download original-model.bin
                You can use the following command to do that:

                wget -O distil-small-en-original-model.bin https://huggingface.co/distil-whisper/distil-small.en/resolve/main/original-model.bin
            """
            )
        model = whisper.load_model(filename)
    elif name == "medium-aishell":
        filename = "./medium-aishell.pt"
        if not Path(filename).is_file():
            raise ValueError(
                """
                Please go to https://huggingface.co/yuekai/icefall_asr_aishell_whisper/tree/main/exp_medium
                to download whisper-medium-aishell1-epoch-10-avg-4.pt
                You can use the following command to do that:

                wget -O medium-aishell.pt https://huggingface.co/yuekai/icefall_asr_aishell_whisper/resolve/main/exp_medium/whisper-medium-aishell1-epoch-10-avg-4.pt
            """
            )
        model = whisper.load_model(filename)
    else:
        model = whisper.load_model(name)

    num_params = sum(p.numel() for p in model.parameters())
    num_encoder_params = sum(p.numel() for p in model.encoder.parameters())
    num_decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"{name} model parameters: {num_params} (or {num_params/1000/1000} M)")
    print(
        f"{name} encoder parameters: {num_encoder_params} (or {num_encoder_params/1000/1000} M)"
    )
    print(
        f"{name} decoder parameters: {num_decoder_params} (or {num_decoder_params/1000/1000} M)"
    )

    convert_tokens(name=name, model=model)

    # write tokens

    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages
    )

    model.eval()
    print(model.dims)
    audio = torch.rand(16000 * 2)
    audio = whisper.pad_or_trim(audio)
    assert audio.shape == (16000 * 30,), audio.shape

    if args.model in ("distil-large-v3", "distil-large-v3.5"):
        n_mels = 128
    elif args.model in (
        "large",
        "large-v3",
        "turbo",
    ):
        n_mels = 128
    else:
        n_mels = 80

    mel = (
        whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(model.device).unsqueeze(0)
    )
    batch_size = 1
    assert mel.shape == (batch_size, n_mels, 30 * 100), mel.shape

    encoder = AudioEncoderTensorCache(model.encoder, model.decoder)

    cross_kv_pair = encoder(mel)
    assert len(cross_kv_pair) == model.dims.n_text_layer, (
        len(cross_kv_pair),
        model.dims.n_text_layer,
    )

    output_names = []
    for i in range(model.dims.n_text_layer):
        k = f"cross_k_{i}"
        v = f"cross_v_{i}"
        output_names.append(k)
        output_names.append(v)

    encoder_filename = f"{name}-encoder.onnx"
    torch.onnx.export(
        encoder,
        mel,
        encoder_filename,
        opset_version=opset_version,
        input_names=[f"{name}-mel"],
        output_names=output_names,
    )

    encoder_meta_data = {
        "model_type": f"whisper-{name}",
        "version": "1",
        "maintainer": "k2-fsa",
        "n_mels": model.dims.n_mels,
        "n_audio_ctx": model.dims.n_audio_ctx,
        "n_audio_state": model.dims.n_audio_state,
        "n_audio_head": model.dims.n_audio_head,
        "n_audio_layer": model.dims.n_audio_layer,
        "n_vocab": model.dims.n_vocab,
        "n_text_ctx": model.dims.n_text_ctx,
        "n_text_state": model.dims.n_text_state,
        "n_text_head": model.dims.n_text_head,
        "n_text_layer": model.dims.n_text_layer,
        "sot_sequence": ",".join(list(map(str, tokenizer.sot_sequence))),
        "all_language_tokens": ",".join(
            list(map(str, tokenizer.all_language_tokens))
        ),  # a list of ids
        "all_language_codes": ",".join(
            tokenizer.all_language_codes
        ),  # e.g., en, de, zh, fr
        "sot": tokenizer.sot,
        "sot_index": tokenizer.sot_sequence.index(tokenizer.sot),
        "eot": tokenizer.eot,
        "blank_id": tokenizer.encode(" ")[0],
        "is_multilingual": int(model.is_multilingual),
        "no_speech": tokenizer.no_speech,
        "non_speech_tokens": ",".join(list(map(str, tokenizer.non_speech_tokens))),
        "transcribe": tokenizer.transcribe,
        "translate": tokenizer.translate,
        "sot_prev": tokenizer.sot_prev,
        "sot_lm": tokenizer.sot_lm,
        "no_timestamps": tokenizer.no_timestamps,
    }
    print(f"encoder_meta_data: {encoder_meta_data}")
    #  add_meta_data(filename=encoder_filename, meta_data=encoder_meta_data)

    tokens = torch.tensor([[tokenizer.sot]], dtype=torch.int32)
    decoder = TextDecoderTensorCache(model.decoder, model.dims.n_text_ctx)

    self_kv_pair = []
    batch_size = 1
    for i in range(model.dims.n_text_layer):
        k = torch.zeros(batch_size, model.dims.n_text_ctx, model.dims.n_text_state)
        v = torch.zeros(batch_size, model.dims.n_text_ctx, model.dims.n_text_state)
        self_kv_pair.append((k, v))

    offset = torch.zeros(1, dtype=torch.int64)
    mask = causal_mask_1d(offset.item(), model.dims.n_text_ctx)

    logits, this_self_kv_pair = decoder(
        tokens,
        self_kv_pair,
        cross_kv_pair,
        offset,
        mask,
    )

    assert logits.shape == (batch_size, tokens.shape[1], model.dims.n_vocab)
    assert len(this_self_kv_pair) == model.dims.n_text_layer, (
        len(this_self_kv_pair),
        model.dims.n_text_layer,
    )

    input_names = [f"{name}-tokens"]
    for i in range(model.dims.n_text_layer):
        k = f"{name}-self_k_{i}"
        v = f"{name}-self_v_{i}"
        input_names.append(k)
        input_names.append(v)

    for i in range(model.dims.n_text_layer):
        k = f"{name}-cross_k_{i}"
        v = f"{name}-cross_v_{i}"
        input_names.append(k)
        input_names.append(v)
    input_names.append(f"{name}-offset")
    input_names.append(f"{name}-mask")

    output_names = [f"{name}-logits"]
    for i in range(model.dims.n_text_layer):
        k = f"{name}-this_self_k_{i}"
        v = f"{name}-this_self_v_{i}"
        output_names.append(k)
        output_names.append(v)

    decoder_filename = f"{name}-decoder.onnx"
    torch.onnx.export(
        decoder,
        (
            tokens,
            self_kv_pair,
            cross_kv_pair,
            offset,
            mask,
        ),
        decoder_filename,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
    )
    return

    if "large" in args.model:
        decoder_external_filename = decoder_filename.split(".onnx")[0]
        decoder_model = onnx.load(decoder_filename)
        onnx.save(
            decoder_model,
            decoder_filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=decoder_external_filename + ".weights",
        )

    # Generate int8 quantization models
    # See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection

    print("Generate int8 quantization models")

    encoder_filename_int8 = f"{name}-encoder.int8.onnx"
    quantize_dynamic(
        model_input=encoder_filename,
        model_output=encoder_filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    decoder_filename_int8 = f"{name}-decoder.int8.onnx"
    quantize_dynamic(
        model_input=decoder_filename,
        model_output=decoder_filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # To fix
    # TypeError: scaled_dot_product_attention(): argument 'is_causal' must be bool, not Tensor
    # See also https://github.com/k2-fsa/sherpa-onnx/issues/1764
    from whisper.model import disable_sdpa

    with disable_sdpa():
        main()
