#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse

import nemo.collections.asr as nemo_asr
import nemo.collections.asr.modules.conformer_encoder as conformer_mod
import torch
import torch.nn as nn


def patched_forward_internal(
    self,
    audio_signal,
    length,
    cache_last_channel=None,
    cache_last_time=None,
    cache_last_channel_len=None,
    bypass_pre_encode=False,
):
    # Patched inference-only version of ConformerEncoder.forward_internal for QNN export.
    #
    # Based on the original at:
    #   ../nemo/nemo/collections/asr/modules/conformer_encoder.py:636
    #
    # Three changes from the original NeMo implementation:
    #
    # 1. Replaced `torch.neg(cache_last_channel_len) + cache_len` with
    #    `cache_len - cache_last_channel_len`. They are mathematically equivalent,
    #    but the original produces a Neg op in the ONNX graph which the Qualcomm
    #    QNN converter (qnn-onnx-converter) does not support, causing:
    #      [ ERROR ] OpConfig validation failed for ElementWiseUnary
    #      [ ERROR ] QnnModel::addNode() validating node node_neg failed.
    #
    # 2. Cast attention masks from bool to float before slicing, then back to
    #    bool after slicing. QNN's StridedSlice does not support boolean tensors,
    #    causing:
    #      [ ERROR ] OpConfig validation failed for StridedSlice
    #      [ ERROR ] QnnModel::addNode() validating node node_slice_3 failed.
    #
    # 3. Removed training-only code paths (stochastic depth, interctc loss capture,
    #    random att_context_size selection) since this is only used during export
    #    where the model is always in eval/inference mode.
    if length is None:
        length = audio_signal.new_full(
            (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
        )

    cur_att_context_size = self.att_context_size

    if not bypass_pre_encode:
        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
            length = length.to(torch.int64)
            if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

        if self.reduction_position is not None and cache_last_channel is not None:
            raise ValueError("Caching with reduction feature is not supported yet!")

    max_audio_length = audio_signal.size(1)
    if cache_last_channel is not None:
        cache_len = self.streaming_cfg.last_channel_cache_size
        cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
        max_audio_length = max_audio_length + cache_len
        padding_length = length + cache_len
        # Original: offset = torch.neg(cache_last_channel_len) + cache_len
        # Changed to subtraction to avoid the Neg op that QNN rejects (see docstring).
        offset = cache_len - cache_last_channel_len
    else:
        padding_length = length
        cache_last_channel_next = None
        cache_len = 0
        offset = None

    if self.self_attention_model == 'rope':
        if self.xscale:
            audio_signal = audio_signal * self.xscale
        audio_signal = self.dropout_pre_encoder(audio_signal)
        pos_emb = None
    else:
        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

    pad_mask, att_mask = self._create_masks(
        att_context_size=cur_att_context_size,
        padding_length=padding_length,
        max_audio_length=max_audio_length,
        offset=offset,
        device=audio_signal.device,
    )

    # Cast masks from bool to float so that QNN StridedSlice can handle them.
    # QNN does not support StridedSlice on boolean tensors.
    # Cast back to bool before passing to layers below.
    pad_mask = pad_mask.to(torch.float32)
    if att_mask is not None:
        att_mask = att_mask.to(torch.float32)

    if cache_last_channel is not None:
        pad_mask = pad_mask[:, cache_len:]
        if att_mask is not None:
            att_mask = att_mask[:, cache_len:]
        cache_last_time_next = []
        cache_last_channel_next = []

    # Cast masks back to bool for the attention layers.
    pad_mask = pad_mask.to(torch.bool)
    if att_mask is not None:
        att_mask = att_mask.to(torch.bool)

    for lth, layer in enumerate(self.layers):
        if cache_last_channel is not None:
            cache_last_channel_cur = cache_last_channel[lth]
            cache_last_time_cur = cache_last_time[lth]
        else:
            cache_last_channel_cur = None
            cache_last_time_cur = None
        audio_signal = layer(
            x=audio_signal,
            att_mask=att_mask,
            pos_emb=pos_emb,
            pad_mask=pad_mask,
            cache_last_channel=cache_last_channel_cur,
            cache_last_time=cache_last_time_cur,
        )

        if cache_last_channel_cur is not None:
            (audio_signal, cache_last_channel_cur, cache_last_time_cur) = audio_signal
            cache_last_channel_next.append(cache_last_channel_cur)
            cache_last_time_next.append(cache_last_time_cur)

        if self.reduction_position == lth:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)
            max_audio_length = audio_signal.size(1)
            if self.self_attention_model != 'rope':
                _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
            pad_mask, att_mask = self._create_masks(
                att_context_size=cur_att_context_size,
                padding_length=length,
                max_audio_length=max_audio_length,
                offset=offset,
                device=audio_signal.device,
            )

    if self.out_proj is not None:
        audio_signal = self.out_proj(audio_signal)

    if self.reduction_position == -1:
        audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

    audio_signal = torch.transpose(audio_signal, 1, 2)
    length = length.to(dtype=torch.int64)

    if cache_last_channel is not None:
        cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
        cache_last_time_next = torch.stack(cache_last_time_next, dim=0)
        return (
            audio_signal,
            length,
            cache_last_channel_next,
            cache_last_time_next,
            torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
        )
    else:
        return audio_signal, length


def patched_streaming_post_process(self, rets, keep_all_outputs=True):
    if len(rets) == 2:
        return rets[0], rets[1], None, None, None

    (
        encoded,
        encoded_len,
        cache_last_channel_next,
        cache_last_time_next,
        cache_last_channel_next_len,
    ) = rets

    if (
        cache_last_channel_next is not None
        and self.streaming_cfg.last_channel_cache_size > 0
    ):
        cache_size = int(self.streaming_cfg.last_channel_cache_size)
        total_size = cache_last_channel_next.shape[2]
        start_idx = total_size - cache_size
        cache_last_channel_next = cache_last_channel_next.narrow(
            2, start_idx, cache_size
        )

    if self.streaming_cfg.valid_out_len > 0 and (
        not keep_all_outputs or self.att_context_style == "regular"
    ):
        valid_len = int(self.streaming_cfg.valid_out_len)
        encoded = encoded.narrow(2, 0, valid_len)
        encoded_len = torch.clamp(encoded_len, max=valid_len)

    return (
        encoded,
        encoded_len,
        cache_last_channel_next,
        cache_last_time_next,
        cache_last_channel_next_len,
    )


# Monkey-patch NeMo's ConformerEncoder to produce an ONNX graph that
# QNN can consume. forward_internal avoids unsupported Neg and
# StridedSlice-on-bool ops; streaming_post_process simplifies the
# cache/output handling.
conformer_mod.ConformerEncoder.forward_internal = patched_forward_internal
conformer_mod.ConformerEncoder.streaming_post_process = patched_streaming_post_process


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--chunk-size-ms",
        type=int,
        choices=[80, 160, 560, 1120],
        required=True,
    )

    parser.add_argument(
        "--model-id",
        type=str,
        help="e.g., nvidia/nemotron-speech-streaming-en-0.6b",
        required=True,
    )
    return parser.parse_args()


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_init_states(self):
        cache_last_channel, cache_last_time, cache_last_channel_len = (
            self.model.encoder.get_initial_cache_state()
        )

        print(cache_last_channel.shape)
        print(cache_last_time.shape)
        print(cache_last_channel_len.shape, cache_last_channel_len)

        return (
            cache_last_channel.transpose(0, 1),
            cache_last_time.transpose(0, 1),
            cache_last_channel_len.to(torch.int32),
        )

    def forward(
        self,
        x: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ):
        """
        Args:
          x: feature tensor, (N, C, T)
          cache_last_channel,
          cache_last_time,
          cache_last_channel_len,

        Returns:
          encoder_out: (N, C, T)
        """
        x_lens = torch.tensor([x.shape[2]], dtype=torch.int64)
        encoder_out, encoder_out_len, *states = self.model.encoder.forward_for_export(
            audio_signal=x,
            length=x_lens,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

        return encoder_out, states


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_init_states(self):
        h, c = self.model.decoder.initialize_state(
            torch.tensor([[1.0]], dtype=torch.float)
        )

        return h, c

    def forward(self, y: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """
        y:  torch.int32, shape (1, 1)
        """
        transcript_len = torch.tensor([1], dtype=torch.int32)
        decoder_out, target_lengths, states = self.model.decoder(
            targets=y, target_length=transcript_len, states=[h, c]
        )
        decoder_out = decoder_out.permute(0, 2, 1)  # (N, C, 1) -> (N, 1, C)
        return decoder_out, states[0], states[1]


class JoinerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor):
        """
        encoder_out: feature tensor, (1, 1, encoder_dim)
        decoder_out: feature tensor, (1, 1, decoder_dim)
        """
        log_probs = self.model.joint.joint(encoder_out, decoder_out)

        return log_probs


def export_encoder(asr_model, window_size, window_shift):
    m = EncoderWrapper(asr_model)
    m.eval()

    feat_dim = asr_model.cfg.preprocessor.features

    states = m.get_init_states()
    x = torch.rand(1, feat_dim, window_size, dtype=torch.float32)

    print("x", x.shape)

    torch.onnx.export(
        m,
        (x, *states),
        "encoder.onnx",
        opset_version=13,
        input_names=[
            f"x_{window_size}_{window_shift}",
            "cache_last_channel",
            "cache_last_time",
            "cache_last_channel_len",
        ],
        output_names=[
            "encoder_out",
            "next_cache_last_channel",
            "next_cache_last_time",
            "next_cache_last_channel_len",
        ],
    )


def export_decoder(asr_model):
    m = DecoderWrapper(asr_model)
    m.eval()

    h, c = m.get_init_states()
    y = torch.tensor([[1]], dtype=torch.int32)

    torch.onnx.export(
        m,
        (y, h, c),
        "decoder.onnx",
        opset_version=13,
        input_names=["y", "h", "c"],
        output_names=["decoder_out", "next_h", "next_c"],
    )


def export_joiner(asr_model, encoder_dim, decoder_dim):
    m = JoinerWrapper(asr_model)
    m.eval()
    encoder_out = torch.rand(1, 1, encoder_dim, dtype=torch.float32)
    decoder_out = torch.rand(1, 1, decoder_dim, dtype=torch.float32)

    torch.onnx.export(
        m,
        (encoder_out, decoder_out),
        "joiner.onnx",
        opset_version=13,
        input_names=["encoder_out", "decoder_out"],
        output_names=["log_probs"],
    )


@torch.no_grad()
def main():
    args = get_args()
    print(vars(args))

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model_id)
    asr_model.eval()

    asr_model.set_export_config({"cache_support": True})

    asr_model.decoder._prepare_for_export()
    asr_model.joint._prepare_for_export()

    chunk_size_ms = args.chunk_size_ms
    chunk_size = chunk_size_ms // 80 - 1
    print("chunk_size", chunk_size)
    asr_model.encoder.set_default_att_context_size([70, chunk_size])
    print("streaming_cfg", asr_model.encoder.streaming_cfg)

    if isinstance(asr_model.encoder.streaming_cfg.pre_encode_cache_size, list):
        pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
    else:
        pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size

    if isinstance(asr_model.encoder.streaming_cfg.chunk_size, list):
        chunk_size = asr_model.encoder.streaming_cfg.chunk_size[1]
    else:
        chunk_size = asr_model.encoder.streaming_cfg.chunk_size

    window_size = chunk_size + pre_encode_cache_size

    window_shift = chunk_size

    print(window_size, window_shift)

    print(type(asr_model))
    print(asr_model)
    print(asr_model.cfg)
    print(asr_model.decoding)
    print(asr_model.decoding.decoding)
    print(asr_model.decoding.decoding._blank_index)
    print(asr_model.decoding.decoding._SOS)

    print("num layers", asr_model.decoder.pred_rnn_layers)

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")

    feat_dim = asr_model.cfg["preprocessor"]["features"]
    print("feat_dim", feat_dim)

    export_encoder(asr_model, window_size=window_size, window_shift=window_shift)
    export_decoder(asr_model)
    export_joiner(
        asr_model,
        encoder_dim=asr_model.joint.enc.in_features,
        decoder_dim=asr_model.joint.pred.in_features,
    )


if __name__ == "__main__":
    main()
