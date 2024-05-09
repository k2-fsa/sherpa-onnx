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


def display(sess):
    print("==========Input==========")
    for i in sess.get_inputs():
        print(i)
    print("==========Output==========")
    for i in sess.get_outputs():
        print(i)


"""
encoder
==========Input==========
NodeArg(name='audio_signal', type='tensor(float)', shape=['audio_signal_dynamic_axes_1', 80, 'audio_signal_dynamic_axes_2'])
NodeArg(name='length', type='tensor(int64)', shape=['length_dynamic_axes_1'])
==========Output==========
NodeArg(name='outputs', type='tensor(float)', shape=['outputs_dynamic_axes_1', 512, 'outputs_dynamic_axes_2'])
NodeArg(name='encoded_lengths', type='tensor(int64)', shape=['encoded_lengths_dynamic_axes_1'])

decoder
==========Input==========
NodeArg(name='targets', type='tensor(int32)', shape=['targets_dynamic_axes_1', 'targets_dynamic_axes_2'])
NodeArg(name='target_length', type='tensor(int32)', shape=['target_length_dynamic_axes_1'])
NodeArg(name='states.1', type='tensor(float)', shape=[1, 'states.1_dim_1', 640])
NodeArg(name='onnx::LSTM_3', type='tensor(float)', shape=[1, 1, 640])
==========Output==========
NodeArg(name='outputs', type='tensor(float)', shape=['outputs_dynamic_axes_1', 640, 'outputs_dynamic_axes_2'])
NodeArg(name='prednet_lengths', type='tensor(int32)', shape=['prednet_lengths_dynamic_axes_1'])
NodeArg(name='states', type='tensor(float)', shape=[1, 'states_dynamic_axes_1', 640])
NodeArg(name='74', type='tensor(float)', shape=[1, 'LSTM74_dim_1', 640])

joiner
==========Input==========
NodeArg(name='encoder_outputs', type='tensor(float)', shape=['encoder_outputs_dynamic_axes_1', 512, 'encoder_outputs_dynamic_axes_2'])
NodeArg(name='decoder_outputs', type='tensor(float)', shape=['decoder_outputs_dynamic_axes_1', 640, 'decoder_outputs_dynamic_axes_2'])
==========Output==========
NodeArg(name='outputs', type='tensor(float)', shape=['outputs_dynamic_axes_1', 'outputs_dynamic_axes_2', 'outputs_dynamic_axes_3', 1025])
"""


class OnnxModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
        joiner: str,
    ):
        self.init_encoder(encoder)
        display(self.encoder)
        self.init_decoder(decoder)
        display(self.decoder)
        self.init_joiner(joiner)
        display(self.joiner)

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
        self.normalize_type = meta["normalize_type"]
        print(meta)

        self.pred_rnn_layers = int(meta["pred_rnn_layers"])
        self.pred_hidden = int(meta["pred_hidden"])

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

    def run_encoder(self, x: np.ndarray):
        # x: (T, C)
        x = torch.from_numpy(x)
        x = x.t().unsqueeze(0)
        # x: [1, C, T]
        x_lens = torch.tensor([x.shape[-1]], dtype=torch.int64)

        (encoder_out, out_len) = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x.numpy(),
                self.encoder.get_inputs()[1].name: x_lens.numpy(),
            },
        )
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

    blank = len(id2token) - 1
    ans = [blank]
    state0, state1 = model.get_decoder_state()
    decoder_out, state0_next, state1_next = model.run_decoder(ans[-1], state0, state1)

    features = compute_features(audio, fbank)
    if model.normalize_type != "":
        assert model.normalize_type == "per_feature", model.normalize_type
        features = torch.from_numpy(features)
        mean = features.mean(dim=1, keepdims=True)
        stddev = features.std(dim=1, keepdims=True)
        features = (features - mean) / stddev
        features = features.numpy()
    print(audio.shape)
    print("features.shape", features.shape)

    encoder_out = model.run_encoder(features)
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
    print(ans)
    tokens = [id2token[i] for i in ans]
    underline = "▁"
    #  underline = b"\xe2\x96\x81".decode()
    text = "".join(tokens).replace(underline, " ").strip()
    print(args.wav)
    print(text)


main()
