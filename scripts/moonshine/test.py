#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)
import datetime as dt

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf


def display(sess, name):
    print(f"=========={name} Input==========")
    for i in sess.get_inputs():
        print(i)
    print(f"=========={name} Output==========")
    for i in sess.get_outputs():
        print(i)


class OnnxModel:
    def __init__(
        self,
        preprocess: str,
        encode: str,
        uncached_decode: str,
        cached_decode: str,
    ):
        self.init_preprocess(preprocess)
        display(self.preprocess, "preprocess")

        self.init_encode(encode)
        display(self.encode, "encode")

        self.init_uncached_decode(uncached_decode)
        display(self.uncached_decode, "uncached_decode")

        self.init_cached_decode(cached_decode)
        display(self.cached_decode, "cached_decode")

    def init_preprocess(self, preprocess):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.preprocess = ort.InferenceSession(
            preprocess,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_encode(self, encode):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.encode = ort.InferenceSession(
            encode,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_uncached_decode(self, uncached_decode):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.uncached_decode = ort.InferenceSession(
            uncached_decode,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_cached_decode(self, cached_decode):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.cached_decode = ort.InferenceSession(
            cached_decode,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    def run_preprocess(self, audio):
        """
        Args:
          audio: (batch_size, num_samples), float32
        Returns:
          A tensor of shape (batch_size, T, dim), float32
        """
        return self.preprocess.run(
            [
                self.preprocess.get_outputs()[0].name,
            ],
            {
                self.preprocess.get_inputs()[0].name: audio,
            },
        )[0]

    def run_encode(self, features):
        """
        Args:
          features: (batch_size, T, dim)
        Returns:
          A tensor of shape (batch_size, T, dim)
        """
        features_len = np.array([features.shape[1]], dtype=np.int32)

        return self.encode.run(
            [
                self.encode.get_outputs()[0].name,
            ],
            {
                self.encode.get_inputs()[0].name: features,
                self.encode.get_inputs()[1].name: features_len,
            },
        )[0]

    def run_uncached_decode(self, token: int, token_len: int, encoder_out: np.ndarray):
        """
        Args:
          token: The current token
          token_len: Number of predicted tokens so far
          encoder_out: A tensor fo shape (batch_size, T, dim)
        Returns:
          A a tuple:
            - a tensor of shape (batch_size, 1, dim)
            - a list of states
        """
        token_tensor = np.array([[token]], dtype=np.int32)
        token_len_tensor = np.array([token_len], dtype=np.int32)

        num_outs = len(self.uncached_decode.get_outputs())
        out_names = [
            self.uncached_decode.get_outputs()[i].name for i in range(num_outs)
        ]

        out = self.uncached_decode.run(
            out_names,
            {
                self.uncached_decode.get_inputs()[0].name: token_tensor,
                self.uncached_decode.get_inputs()[1].name: encoder_out,
                self.uncached_decode.get_inputs()[2].name: token_len_tensor,
            },
        )

        logits = out[0]
        states = out[1:]

        return logits, states

    def run_cached_decode(
        self, token: int, token_len: int, encoder_out: np.ndarray, states
    ):
        """
        Args:
          token: The current token
          token_len: Number of predicted tokens so far
          encoder_out: A tensor of shape (batch_size, T, dim)
          states: previous states
        Returns:
          A a tuple:
            - a tensor of shape (batch_size, 1, dim)
            - a list of states
        """
        token_tensor = np.array([[token]], dtype=np.int32)
        token_len_tensor = np.array([token_len], dtype=np.int32)

        num_outs = len(self.cached_decode.get_outputs())
        out_names = [self.cached_decode.get_outputs()[i].name for i in range(num_outs)]

        states_inputs = {}
        for i in range(3, len(self.cached_decode.get_inputs())):
            name = self.cached_decode.get_inputs()[i].name
            states_inputs[name] = states[i - 3]

        out = self.cached_decode.run(
            out_names,
            {
                self.cached_decode.get_inputs()[0].name: token_tensor,
                self.cached_decode.get_inputs()[1].name: encoder_out,
                self.cached_decode.get_inputs()[2].name: token_len_tensor,
                **states_inputs,
            },
        )

        logits = out[0]
        states = out[1:]

        return logits, states


def main():
    wave = "./1.wav"
    id2token = dict()
    token2id = dict()
    with open("./tokens.txt", encoding="utf-8") as f:
        for k, line in enumerate(f):
            t, idx = line.split("\t")
            id2token[int(idx)] = t
            token2id[t] = int(idx)

    model = OnnxModel(
        preprocess="./preprocess.onnx",
        encode="./encode.int8.onnx",
        uncached_decode="./uncached_decode.int8.onnx",
        cached_decode="./cached_decode.int8.onnx",
    )

    audio, sample_rate = sf.read(wave, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel
    if sample_rate != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )
        sample_rate = 16000
    audio = audio[None]  # (1, num_samples)
    print("audio.shape", audio.shape)  # (1, 159414)

    start_t = dt.datetime.now()

    features = model.run_preprocess(audio)  # (1, 413, 288)
    print("features", features.shape)

    sos = token2id["<s>"]
    eos = token2id["</s>"]

    tokens = [sos]

    encoder_out = model.run_encode(features)
    print("encoder_out.shape", encoder_out.shape)  # (1, 413, 288)

    logits, states = model.run_uncached_decode(
        token=tokens[-1],
        token_len=len(tokens),
        encoder_out=encoder_out,
    )

    print("logits.shape", logits.shape)  # (1, 1, 32768)
    print("len(states)", len(states))  # 24

    max_len = int((audio.shape[-1] / 16000) * 6)

    for i in range(max_len):
        token = logits.squeeze().argmax()
        if token == eos:
            break
        tokens.append(token)

        logits, states = model.run_cached_decode(
            token=tokens[-1],
            token_len=len(tokens),
            encoder_out=encoder_out,
            states=states,
        )

    tokens = tokens[1:]  # remove sos
    words = [id2token[i] for i in tokens]
    underline = "▁"
    #  underline = b"\xe2\x96\x81".decode()
    text = "".join(words).replace(underline, " ").strip()

    end_t = dt.datetime.now()
    t = (end_t - start_t).total_seconds()
    rtf = t * 16000 / audio.shape[-1]

    print(text)
    print("RTF:", rtf)


if __name__ == "__main__":
    main()
