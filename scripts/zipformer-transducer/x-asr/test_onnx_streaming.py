#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--use-int8",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--wav",
        type=str,
        required=True,
    )
    return parser.parse_args()


def show_sess(sess, hint):
    print(f"---{hint} input---")
    for i in sess.get_inputs():
        print(i)

    print(f"---{hint} output---")

    for i in sess.get_outputs():
        print(i)
    print("=" * 5)
    print()


def load_tokens(filename):
    ans = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            t, i = line.strip().split()
            ans[int(i)] = t
            pass

    return ans


def load_audio(filename: str):
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel

    if sample_rate != 16000:
        import librosa

        data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def load_model(use_int8):
    if use_int8:
        model = OnnxModel(
            encoder="./encoder.int8.onnx",
            decoder="./decoder.onnx",
            joiner="./joiner.int8.onnx",
        )
    else:
        model = OnnxModel(
            encoder="./encoder.onnx",
            decoder="./decoder.onnx",
            joiner="./joiner.onnx",
        )
    return model


def compute_feat(
    samples: np.ndarray,
    sample_rate: int,
):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, samples.tolist())
    online_fbank.input_finished()

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )

    features = np.pad(
        features,
        ((0, 100), (0, 0)),  # pad 100 frames, e.g., 1 second
        mode="constant",
        constant_values=0,
    )

    features = np.ascontiguousarray(features)

    assert features.data.contiguous is True
    assert features.dtype == np.float32, features.dtype

    return features


class OnnxModel:
    def __init__(self, encoder, decoder, joiner):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        self.joiner = ort.InferenceSession(
            joiner,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        #  show_sess(self.encoder, "encoder")
        #  show_sess(self.decoder, "decoder")
        #  show_sess(self.joiner, "joiner")

        encoder_meta = self.encoder.get_modelmeta().custom_metadata_map
        print(encoder_meta)

        model_type = encoder_meta["model_type"]
        assert model_type == "zipformer2", model_type

        self.decode_chunk_len = int(encoder_meta["decode_chunk_len"])
        self.T = int(encoder_meta["T"])

        decoder_meta = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(decoder_meta["context_size"])
        self.vocab_size = int(decoder_meta["vocab_size"])

    def get_encoder_states(self):
        states = []
        for n in self.encoder.get_inputs()[1:]:
            assert n.type in ("tensor(float)", "tensor(int64)", "tensor(int32)"), n
            if n.type == "tensor(float)":
                dtype = np.float32
            elif n.type == "tensor(int64)":
                dtype = np.int64
            else:
                dtype = np.int32
            shape = [1 if isinstance(s, str) else s for s in n.shape]
            s = np.zeros(shape, dtype=dtype)
            states.append(s)
        return states

    def run_encoder(self, x, states):
        d = dict()
        for i, n in enumerate(self.encoder.get_inputs()):
            if i == 0:
                d[n.name] = x
            else:
                d[n.name] = states[i - 1]
        out_names = [n.name for n in self.encoder.get_outputs()]

        out = self.encoder.run(out_names, d)
        return out[0], out[1:]

    def run_decoder(self, hyp):

        n = self.decoder.get_inputs()[0]
        assert n.type in ("tensor(int64)", "tensor(int32)"), n
        if n.type == "tensor(int64)":
            dtype = np.int64
        else:
            dtype = np.int32

        hyp = np.array([hyp], dtype=dtype)
        out = self.decoder.run(
            [self.decoder.get_outputs()[0].name],
            {self.decoder.get_inputs()[0].name: hyp},
        )
        return out[0]

    def run_joiner(self, encoder_out, decoder_out):
        out = self.joiner.run(
            [self.joiner.get_outputs()[0].name],
            {
                self.joiner.get_inputs()[0].name: encoder_out,
                self.joiner.get_inputs()[1].name: decoder_out,
            },
        )
        return out[0]


def main():
    args = get_args()
    print(vars(args))

    wave = args.wav
    samples, sample_rate = load_audio(wave)

    features = compute_feat(
        samples=samples,
        sample_rate=sample_rate,
    )
    print("features", features.shape)

    id2token = load_tokens("./tokens.txt")

    model = load_model(args.use_int8)
    states = model.get_encoder_states()

    state_name_list = [n.name for n in model.encoder.get_inputs()[1:]]
    assert len(states) == len(state_name_list), (len(states), len(state_name_list))

    blank_id = 0

    hyp = [blank_id] * model.context_size
    decoder_out = model.run_decoder(hyp)

    frame_size = model.T
    frame_shift = model.decode_chunk_len
    start = 0
    while start + frame_size < features.shape[0]:
        x = features[start : start + frame_size]
        start += frame_shift

        x = x[None]
        encoder_out, states = model.run_encoder(x, states)
        num_frames = encoder_out.shape[1]
        for k in range(num_frames):
            cur_encoder_out = encoder_out[0, k : k + 1]
            joiner_out = model.run_joiner(cur_encoder_out, decoder_out)
            token_id = joiner_out.argmax()
            if token_id != blank_id:
                hyp.append(token_id)
                decoder_out = model.run_decoder(hyp[-model.context_size :])
    tokens = [id2token[i] for i in hyp[model.context_size :]]
    print(tokens)
    text = "".join(tokens)
    print(text)


if __name__ == "__main__":
    main()
