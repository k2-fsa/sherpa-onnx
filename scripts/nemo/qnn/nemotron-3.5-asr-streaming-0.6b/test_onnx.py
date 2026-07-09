#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
from pathlib import Path

import kaldi_native_fbank as knf
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

"""
=========./encoder.onnx==========
NodeArg(name='x_121_112', type='tensor(float)', shape=[1, 128, 121])
NodeArg(name='cache_last_channel', type='tensor(float)', shape=[1, 24, 70, 1024])
NodeArg(name='cache_last_time', type='tensor(float)', shape=[1, 24, 1024, 8])
NodeArg(name='cache_last_channel_len', type='tensor(int32)', shape=[1])
-----
NodeArg(name='encoder_out', type='tensor(float)', shape=[1, 1024, 14])
NodeArg(name='next_cache_last_channel', type='tensor(float)', shape=[1, 24, 70, 1024])
NodeArg(name='next_cache_last_time', type='tensor(float)', shape=[1, 24, 1024, 8])
NodeArg(name='next_cache_last_channel_len', type='tensor(int32)', shape=[1])
=========./decoder.onnx==========
NodeArg(name='y', type='tensor(int32)', shape=[1, 1])
NodeArg(name='h', type='tensor(float)', shape=[2, 1, 640])
NodeArg(name='c', type='tensor(float)', shape=[2, 1, 640])
-----
NodeArg(name='decoder_out', type='tensor(float)', shape=[1, 1, 640])
NodeArg(name='next_h', type='tensor(float)', shape=[2, 1, 640])
NodeArg(name='next_c', type='tensor(float)', shape=[2, 1, 640])
=========./joiner.onnx==========
NodeArg(name='encoder_out', type='tensor(float)', shape=[1, 1, 1024])
NodeArg(name='decoder_out', type='tensor(float)', shape=[1, 1, 640])
-----
NodeArg(name='logits', type='tensor(float)', shape=[1, 1, 1, 1025])
"""


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--wav",
        type=str,
        required=True,
    )
    return parser.parse_args()


def create_fbank(feat_dim: int):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.remove_dc_offset = False
    opts.frame_opts.window_type = "hann"

    opts.mel_opts.low_freq = 0
    opts.mel_opts.num_bins = feat_dim

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

    features = np.stack(ans)
    features = np.ascontiguousarray(features)
    return features


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

        x = self.encoder.get_inputs()[0].name
        _, window_size, window_shift = x.split("_")

        self.window_size = int(window_size)
        self.window_shift = int(window_shift)

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

    def get_encoder_state(self):
        cache_last_channel_shape = self.encoder.get_inputs()[1].shape
        cache_last_time_shape = self.encoder.get_inputs()[2].shape
        cache_last_channel_len_shape = self.encoder.get_inputs()[3].shape

        s0 = np.zeros(cache_last_channel_shape, dtype=np.float32)
        s1 = np.zeros(cache_last_time_shape, dtype=np.float32)
        s2 = np.zeros(cache_last_channel_len_shape, dtype=np.int32)

        return s0, s1, s2

    def get_decoder_state(self):
        h_shape = self.decoder.get_inputs()[1].shape
        c_shape = self.decoder.get_inputs()[2].shape

        h = np.zeros(h_shape, dtype=np.float32)
        c = np.zeros(c_shape, dtype=np.float32)
        return h, c

    def run_encoder(self, x: np.ndarray, states, prompt_index):
        encoder_out, *next_states = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
                self.encoder.get_outputs()[2].name,
                self.encoder.get_outputs()[3].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x,
                self.encoder.get_inputs()[1].name: states[0],
                self.encoder.get_inputs()[2].name: states[1],
                self.encoder.get_inputs()[3].name: states[2],
                self.encoder.get_inputs()[4].name: prompt_index,
            },
        )
        return encoder_out, next_states

    def run_decoder(
        self,
        token: int,
        h: np.ndarray,
        c: np.ndarray,
    ):
        y = np.array([[token]], dtype=np.int32)

        (
            decoder_out,
            next_h,
            next_c,
        ) = self.decoder.run(
            [
                self.decoder.get_outputs()[0].name,
                self.decoder.get_outputs()[1].name,
                self.decoder.get_outputs()[2].name,
            ],
            {
                self.decoder.get_inputs()[0].name: y,
                self.decoder.get_inputs()[1].name: h,
                self.decoder.get_inputs()[2].name: c,
            },
        )
        return decoder_out, next_h, next_c

    def run_joiner(
        self,
        encoder_out: np.ndarray,
        decoder_out: np.ndarray,
    ):
        logits = self.joiner.run(
            [
                self.joiner.get_outputs()[0].name,
            ],
            {
                self.joiner.get_inputs()[0].name: encoder_out,
                self.joiner.get_inputs()[1].name: decoder_out,
            },
        )[0]
        return logits


def main():
    args = get_args()
    model = OnnxModel("encoder.onnx", "decoder.onnx", "joiner.onnx")

    wav = args.wav
    name = Path(wav).stem
    window_size = model.encoder.get_inputs()[0].shape[2]
    assert window_size == model.window_size, (window_size, model.window_size)

    feat_dim = model.encoder.get_inputs()[0].shape[1]
    window_shift = model.window_shift

    assert feat_dim in (80, 128), feat_dim

    id2token = dict()
    with open("tokens.txt", encoding="utf-8") as f:
        for line in f:
            t, idx = line.split()
            id2token[int(idx)] = t

    fbank = create_fbank(feat_dim=feat_dim)
    audio, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel
    if sample_rate != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )
        sample_rate = 16000

    tail_padding = np.zeros(sample_rate * 1)

    audio = np.concatenate([audio, tail_padding])

    blank = len(id2token) - 1
    ans = [blank]

    encoder_input_list = []
    decoder_input_list = []
    joiner_input_list = []

    h, c = model.get_decoder_state()

    decoder_input_list.append((ans[-1], h, c))
    decoder_out, h, c = model.run_decoder(ans[-1], h, c)

    features = compute_features(audio, fbank)
    features.tofile(f"{name}-features.raw")

    encoder_states = model.get_encoder_state()
    prompt_index = np.array([101], dtype=np.int32)

    max_symbols_per_frame = 10

    for i in range(0, features.shape[0], window_shift):
        f = features[i : i + window_size]
        if f.shape[0] < window_size:
            break

        encoder_input_list.append([f, encoder_states, prompt_index])
        encoder_out, encoder_states = model.run_encoder(
            f[None].transpose(0, 2, 1), encoder_states, prompt_index
        )
        encoder_out = encoder_out.transpose(0, 2, 1)

        num_symbols = 0
        t = 0
        while t < encoder_out.shape[1]:
            encoder_out_t = encoder_out[:, t : t + 1]

            joiner_input_list.append((encoder_out_t, decoder_out))
            logits = model.run_joiner(encoder_out_t, decoder_out)

            logits = logits.squeeze()
            idx = np.argmax(logits).item()
            if idx != blank:
                ans.append(idx)

                decoder_input_list.append((ans[-1], h, c))
                decoder_out, h, c = model.run_decoder(ans[-1], h, c)

                num_symbols += 1

                if num_symbols > max_symbols_per_frame:
                    num_symbols = 0
                    t += 1
            else:
                t += 1
                num_symbols = 0

    if True:
        with open(f"{name}-encoder.txt", "w") as f:
            for i, (x, states, prompt_index) in enumerate(encoder_input_list):
                x_name = f"{name}-{i}-x.raw"
                x.tofile(x_name)

                s0_name = f"{name}-{i}-s0.raw"
                states[0].tofile(s0_name)

                s1_name = f"{name}-{i}-s1.raw"
                states[1].tofile(s1_name)

                s2_name = f"{name}-{i}-s2.raw"
                states[2].tofile(s2_name)

                prompt_name = f"{name}-{i}-prompt.raw"
                prompt_index.tofile(prompt_name)

                f.write(f"{x_name} {s0_name} {s1_name} {s2_name} {prompt_name}\n")

        with open(f"{name}-decoder.txt", "w") as f:
            for i, (y, h, c) in enumerate(decoder_input_list):
                y_name = f"{name}-{i}-y.raw"
                np.array([y], dtype=np.int32).tofile(y_name)

                h_name = f"{name}-{i}-h.raw"
                h.tofile(h_name)

                c_name = f"{name}-{i}-c.raw"
                c.tofile(c_name)
                f.write(f"{y_name} {h_name} {c_name}\n")

        with open(f"{name}-joiner.txt", "w") as f:
            for i, (e, d) in enumerate(joiner_input_list):
                e_name = f"{name}-{i}-joiner-e.raw"
                e.tofile(e_name)

                d_name = f"{name}-{i}-joiner-d.raw"
                d.tofile(d_name)

                f.write(f"{e_name} {d_name}\n")

    ans = ans[1:]  # remove the first blank
    print(ans)
    tokens = [id2token[i] for i in ans]
    underline = "▁"
    #  underline = b"\xe2\x96\x81".decode()
    text = "".join(tokens).replace(underline, " ").strip()
    print("result")
    print(text)


if __name__ == "__main__":
    main()
