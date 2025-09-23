#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model.onnx",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--wave",
        type=str,
        required=True,
        help="The input wave to be recognized",
    )

    return parser.parse_args()


class OnnxModel:
    def __init__(self, filename):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        meta = self.model.get_modelmeta().custom_metadata_map

        self.frame_length_ms = int(meta["frame_length_ms"])
        self.sample_rate = int(meta["sample_rate"])
        self.state_dim = int(meta["state_dim"])

    def get_init_state(self, batch_size=1):
        return np.zeros((batch_size, self.state_dim), dtype=np.float16)

    def __call__(self, x, state):
        """
        Args:
          x: (batch_size, num_samples, 1), int32
          state: (batch_size, 219729)
        Returns:
          log_probs: (batch_size, num_frames, vocab_size)
          next_state: (batch_size, 219729)
        """
        log_prob, next_state = self.model.run(
            [
                self.model.get_outputs()[0].name,
                self.model.get_outputs()[1].name,
            ],
            {
                self.model.get_inputs()[0].name: x,
                self.model.get_inputs()[1].name: state,
            },
        )
        return log_prob, next_state


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def load_tokens(filename):
    ans = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 1:
                ans[int(fields[0])] = " "
            else:
                ans[int(fields[1])] = fields[0]
    return ans


def compute_feat(
    samples,
    sample_rate,
    frame_length_ms: int,
):
    opts = knf.RawAudioSamplesOptions()
    opts.frame_opts.samp_freq = sample_rate
    opts.frame_opts.frame_length_ms = frame_length_ms
    opts.frame_opts.frame_shift_ms = frame_length_ms

    raw_audio_samples = knf.OnlineRawAudioSamples(opts)

    raw_audio_samples.accept_waveform(sample_rate, samples)
    raw_audio_samples.input_finished()

    features = []

    for i in range(raw_audio_samples.num_frames_ready):
        f = raw_audio_samples.get_frame(i)
        features.append(f)

    return (np.array(features, dtype=np.float32) * 32768).astype(np.int32)


def main():
    args = get_args()
    print(vars(args))

    model = OnnxModel(filename=args.model)

    samples, sample_rate = load_audio(args.wave)
    if sample_rate != model.sample_rate:
        import librosa

        samples = librosa.resample(
            samples, orig_sr=sample_rate, target_sr=model.sample_rate
        )
        sample_rate = model.sample_rate

    # Pad 0.5 seconds
    samples = np.pad(samples, (2400, 2400))

    features = compute_feat(
        samples=samples,
        sample_rate=sample_rate,
        frame_length_ms=model.frame_length_ms,
    )

    id2token = load_tokens(args.tokens)

    blank = -2
    for idx, token in id2token.items():
        if token == "<blk>":
            blank = idx

    state = model.get_init_state()
    token_id_list = []
    for f in features:
        log_probs, state = model(f[None, :, None], state)

        max_token_ids = log_probs[0].argmax(axis=-1).tolist()
        token_id_list += max_token_ids

    unique_ids = []
    prev = -1
    for t in token_id_list:
        if t == blank:
            prev = t
            continue

        if t == prev:
            continue

        prev = t
        unique_ids.append(prev)
    text = "".join([id2token[i] for i in unique_ids])
    print(text)


if __name__ == "__main__":
    main()
