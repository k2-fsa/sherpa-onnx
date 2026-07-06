#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse

import nemo.collections.asr as nemo_asr
import torch


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
            cache_last_channel,
            cache_last_time,
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
        encoder_out, encoder_out_len, *states = self.model.encoder(
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
