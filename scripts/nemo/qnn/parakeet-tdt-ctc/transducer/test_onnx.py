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
==========encoder input==========
NodeArg(name='x', type='tensor(float)', shape=[1, 80, 1000])
==========encoder output==========
NodeArg(name='encoder_out', type='tensor(float)', shape=[1, 512, 125])

==========decoder input==========
NodeArg(name='y', type='tensor(int32)', shape=[1, 1])
NodeArg(name='h', type='tensor(float)', shape=[1, 1, 640])
NodeArg(name='c', type='tensor(float)', shape=[1, 1, 640])
==========decoder output==========
NodeArg(name='decoder_out', type='tensor(float)', shape=[1, 1, 640])
NodeArg(name='next_h', type='tensor(float)', shape=[1, 1, 640])
NodeArg(name='next_c', type='tensor(float)', shape=[1, 1, 640])

==========joiner input==========
NodeArg(name='encoder_out', type='tensor(float)', shape=[1, 1, 512])
NodeArg(name='decoder_out', type='tensor(float)', shape=[1, 1, 640])
==========joiner output==========
NodeArg(name='log_probs', type='tensor(float)', shape=[1, 1, 1, 1030])
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


def compute_features(audio, fbank, max_len=-1):
    assert len(audio.shape) == 1, audio.shape
    fbank.accept_waveform(16000, audio)
    ans = []
    processed = 0
    while processed < fbank.num_frames_ready:
        ans.append(np.array(fbank.get_frame(processed)))
        processed += 1

    features = np.stack(ans)

    if max_len > 0:
        if features.shape[0] > max_len:
            features = features[:max_len]
        elif features.shape[0] < max_len:
            features = np.pad(
                features,
                ((0, max_len - features.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )

    features = np.ascontiguousarray(features)
    return features


def display(sess, name):
    print(f"=========={name} input==========")
    for i in sess.get_inputs():
        print(i)
    print(f"=========={name} output==========")
    for i in sess.get_outputs():
        print(i)
    print()


class OnnxModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
        joiner: str,
    ):
        self.init_encoder(encoder)
        display(self.encoder, "encoder")

        self.init_decoder(decoder)
        display(self.decoder, "decoder")

        self.init_joiner(joiner)
        display(self.joiner, "joiner")

    def init_encoder(self, encoder):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

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
        h_shape = self.decoder.get_inputs()[1].shape
        c_shape = self.decoder.get_inputs()[2].shape

        h = np.zeros(h_shape, dtype=np.float32)
        c = np.zeros(c_shape, dtype=np.float32)
        return h, c

    def run_encoder(self, x: np.ndarray):
        encoder_out = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x,
            },
        )[0]
        return encoder_out

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
        log_probs = self.joiner.run(
            [
                self.joiner.get_outputs()[0].name,
            ],
            {
                self.joiner.get_inputs()[0].name: encoder_out,
                self.joiner.get_inputs()[1].name: decoder_out,
            },
        )[0]
        return log_probs


def main():
    args = get_args()
    print(vars(args))

    name = Path(args.wav).stem

    model = OnnxModel("encoder.onnx", "decoder.onnx", "joiner.onnx")
    max_len = model.encoder.get_inputs()[0].shape[2]
    feat_dim = model.encoder.get_inputs()[0].shape[1]
    assert feat_dim in (80, 128), feat_dim

    id2token = dict()
    with open("tokens.txt", encoding="utf-8") as f:
        for line in f:
            t, idx = line.split()
            id2token[int(idx)] = t

    fbank = create_fbank(feat_dim=feat_dim)
    audio, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
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

    decoder_input_list = []

    joiner_input_list = []

    h, c = model.get_decoder_state()

    decoder_input_list.append((ans[-1], h, c))
    decoder_out, h, c = model.run_decoder(ans[-1], h, c)

    features = compute_features(audio, fbank, max_len)
    mean = features.mean(axis=0, keepdims=True)
    stddev = features.std(axis=0, keepdims=True) + 1e-5
    features = (features - mean) / stddev
    features.tofile(f"{name}.raw")
    print("features", features.shape)

    encoder_out = model.run_encoder(features[None].transpose(0, 2, 1))
    encoder_out = encoder_out.transpose(0, 2, 1)
    for t in range(encoder_out.shape[1]):
        encoder_out_t = encoder_out[:, t : t + 1]

        joiner_input_list.append((encoder_out_t, decoder_out))
        logits = model.run_joiner(encoder_out_t, decoder_out)

        logits = logits.squeeze()[: blank + 1]
        idx = np.argmax(logits).item()
        if idx != blank:
            ans.append(idx)

            decoder_input_list.append((ans[-1], h, c))
            decoder_out, h, c = model.run_decoder(ans[-1], h, c)

    if True:
        # Don't quantize the decoder
        with open(f"{name}-decoder.txt", "w") as f:
            for i, (y, h, c) in enumerate(decoder_input_list):
                y_name = f"{name}-{i}-y.raw"
                np.array([y], dtype=np.int32).tofile(y_name)

                h_name = f"{name}-{i}-h.raw"
                h.tofile(h_name)

                c_name = f"{name}-{i}-c.raw"
                c.tofile(c_name)
                f.write(f"{y_name} {h_name} {c_name}\n")

        # Don't quantize the joiner
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
