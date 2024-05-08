#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
from pathlib import Path

import kaldi_native_fbank as knf
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder", type=str, required=True, help="Path to encoder.onnx"
    )
    parser.add_argument(
        "--decoder", type=str, required=True, help="Path to decoder.onnx"
    )
    parser.add_argument("--joiner", type=str, required=True, help="Path to joiner.onnx")

    parser.add_argument("--tokens", type=str, required=True, help="Path to tokens.txt")

    parser.add_argument("--wav", type=str, required=True, help="Path to test.wav")

    return parser.parse_args()


def create_fbank():
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.remove_dc_offset = False
    opts.frame_opts.window_type = "hann"

    opts.mel_opts.low_freq = 0
    opts.mel_opts.num_bins = 80

    opts.mel_opts.is_librosa = True

    fbank = knf.OnlineFbank(opts)
    return fbank


def compute_features(audio, fbank):
    assert len(audio.shape) == 1, audio.shape
    fbank.accept_waveform(16000, audio)
    ans = []
    processed = 0
    while processed < fbank.num_frames_ready:
        ans.append(np.array(fbank.get_frame(processed)))
        processed += 1
    ans = np.stack(ans)
    return ans


class OnnxModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
        joiner: str,
    ):
        self.init_encoder(encoder)
        self.init_decoder(decoder)
        self.init_joiner(joiner)

    def init_encoder(self, encoder):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

        meta = self.encoder.get_modelmeta().custom_metadata_map
        print(meta)

        self.window_size = int(meta["window_size"])
        self.chunk_shift = int(meta["chunk_shift"])

        self.cache_last_channel_dim1 = int(meta["cache_last_channel_dim1"])
        self.cache_last_channel_dim2 = int(meta["cache_last_channel_dim2"])
        self.cache_last_channel_dim3 = int(meta["cache_last_channel_dim3"])

        self.cache_last_time_dim1 = int(meta["cache_last_time_dim1"])
        self.cache_last_time_dim2 = int(meta["cache_last_time_dim2"])
        self.cache_last_time_dim3 = int(meta["cache_last_time_dim3"])

        self.pred_rnn_layers = int(meta["pred_rnn_layers"])
        self.pred_hidden = int(meta["pred_hidden"])

        self.init_cache_state()

    def init_decoder(self, decoder):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_joiner(self, joiner):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.joiner = ort.InferenceSession(
            joiner,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    def get_decoder_state(self):
        batch_size = 1
        state0 = torch.zeros(self.pred_rnn_layers, batch_size, self.pred_hidden).numpy()
        state1 = torch.zeros(self.pred_rnn_layers, batch_size, self.pred_hidden).numpy()
        return state0, state1

    def init_cache_state(self):
        self.cache_last_channel = torch.zeros(
            1,
            self.cache_last_channel_dim1,
            self.cache_last_channel_dim2,
            self.cache_last_channel_dim3,
            dtype=torch.float32,
        ).numpy()

        self.cache_last_time = torch.zeros(
            1,
            self.cache_last_time_dim1,
            self.cache_last_time_dim2,
            self.cache_last_time_dim3,
            dtype=torch.float32,
        ).numpy()

        self.cache_last_channel_len = torch.ones([1], dtype=torch.int64).numpy()

    def run_encoder(self, x: np.ndarray):
        # x: (T, C)
        x = torch.from_numpy(x)
        x = x.t().unsqueeze(0)
        # x: [1, C, T]
        x_lens = torch.tensor([x.shape[-1]], dtype=torch.int64)

        (
            encoder_out,
            out_len,
            cache_last_channel_next,
            cache_last_time_next,
            cache_last_channel_len_next,
        ) = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
                self.encoder.get_outputs()[2].name,
                self.encoder.get_outputs()[3].name,
                self.encoder.get_outputs()[4].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x.numpy(),
                self.encoder.get_inputs()[1].name: x_lens.numpy(),
                self.encoder.get_inputs()[2].name: self.cache_last_channel,
                self.encoder.get_inputs()[3].name: self.cache_last_time,
                self.encoder.get_inputs()[4].name: self.cache_last_channel_len,
            },
        )
        self.cache_last_channel = cache_last_channel_next
        self.cache_last_time = cache_last_time_next
        self.cache_last_channel_len = cache_last_channel_len_next

        # [batch_size, dim, T]
        return encoder_out

    def run_decoder(
        self,
        token: int,
        state0: np.ndarray,
        state1: np.ndarray,
    ):
        target = torch.tensor([[token]], dtype=torch.int32).numpy()
        target_len = torch.tensor([1], dtype=torch.int32).numpy()

        (
            decoder_out,
            decoder_out_length,
            state0_next,
            state1_next,
        ) = self.decoder.run(
            [
                self.decoder.get_outputs()[0].name,
                self.decoder.get_outputs()[1].name,
                self.decoder.get_outputs()[2].name,
                self.decoder.get_outputs()[3].name,
            ],
            {
                self.decoder.get_inputs()[0].name: target,
                self.decoder.get_inputs()[1].name: target_len,
                self.decoder.get_inputs()[2].name: state0,
                self.decoder.get_inputs()[3].name: state1,
            },
        )
        return decoder_out, state0_next, state1_next

    def run_joiner(
        self,
        encoder_out: np.ndarray,
        decoder_out: np.ndarray,
    ):
        # encoder_out: [batch_size,  dim, 1]
        # decoder_out: [batch_size,  dim, 1]
        logit = self.joiner.run(
            [
                self.joiner.get_outputs()[0].name,
            ],
            {
                self.joiner.get_inputs()[0].name: encoder_out,
                self.joiner.get_inputs()[1].name: decoder_out,
            },
        )[0]
        # logit: [batch_size, 1, 1, vocab_size]
        return logit


def main():
    args = get_args()
    assert Path(args.encoder).is_file(), args.encoder
    assert Path(args.decoder).is_file(), args.decoder
    assert Path(args.joiner).is_file(), args.joiner
    assert Path(args.tokens).is_file(), args.tokens
    assert Path(args.wav).is_file(), args.wav

    print(vars(args))

    model = OnnxModel(args.encoder, args.decoder, args.joiner)

    id2token = dict()
    with open(args.tokens, encoding="utf-8") as f:
        for line in f:
            t, idx = line.split()
            id2token[int(idx)] = t

    fbank = create_fbank()
    audio, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel
    if sample_rate != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )
        sample_rate = 16000

    tail_padding = np.zeros(sample_rate * 2)

    audio = np.concatenate([audio, tail_padding])

    window_size = model.window_size
    chunk_shift = model.chunk_shift

    blank = len(id2token) - 1
    ans = [blank]
    state0, state1 = model.get_decoder_state()
    decoder_out, state0_next, state1_next = model.run_decoder(ans[-1], state0, state1)

    features = compute_features(audio, fbank)
    num_chunks = (features.shape[0] - window_size) // chunk_shift + 1
    for i in range(num_chunks):
        start = i * chunk_shift
        end = start + window_size
        chunk = features[start:end, :]

        encoder_out = model.run_encoder(chunk)
        # encoder_out:[batch_size, dim, T)
        for t in range(encoder_out.shape[2]):
            encoder_out_t = encoder_out[:, :, t : t + 1]
            logits = model.run_joiner(encoder_out_t, decoder_out)
            logits = torch.from_numpy(logits)
            logits = logits.squeeze()
            idx = torch.argmax(logits, dim=-1).item()
            if idx != blank:
                ans.append(idx)
                state0 = state0_next
                state1 = state1_next
                decoder_out, state0_next, state1_next = model.run_decoder(
                    ans[-1], state0, state1
                )

    ans = ans[1:]  # remove the first blank
    tokens = [id2token[i] for i in ans]
    underline = "▁"
    #  underline = b"\xe2\x96\x81".decode()
    text = "".join(tokens).replace(underline, " ").strip()
    print(args.wav)
    print(text)


main()
