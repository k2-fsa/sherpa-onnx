#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
import onnxruntime as ort
import argparse
import soundfile as sf
from typing import Tuple
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the onnx model",
    )

    parser.add_argument(
        "--wav",
        type=str,
        required=True,
        help="Path to the input wav",
    )
    return parser.parse_args()


class OnnxModel:
    def __init__(
        self,
        model: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1
        self.model = ort.InferenceSession(
            model,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    def get_init_states(self):
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)
        return h, c

    def __call__(self, x, h, c):
        """
        Args:
          x: (1, 512)
          h: (2, 1, 64)
          c: (2, 1, 64)
        Returns:
          prob: (1, 1)
          next_h: (2, 1, 64)
          next_c: (2, 1, 64)
        """
        x = x[None]
        out, next_h, next_c = self.model.run(
            [
                self.model.get_outputs()[0].name,
                self.model.get_outputs()[1].name,
                self.model.get_outputs()[2].name,
            ],
            {
                self.model.get_inputs()[0].name: x,
                self.model.get_inputs()[1].name: h,
                self.model.get_inputs()[2].name: c,
            },
        )
        return out, next_h, next_c


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def main():
    args = get_args()

    samples, sample_rate = load_audio(args.wav)
    if sample_rate != 16000:
        import librosa

        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    model = OnnxModel(args.model)
    probs = []
    h, c = model.get_init_states()
    window_size = 512
    num_windows = samples.shape[0] // window_size
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        p, h, c = model(samples[start:end], h, c)
        probs.append(p[0].item())

    threshold = 0.5
    out = np.array(probs) > threshold
    out = out.tolist()
    min_speech_duration = 0.25 * sample_rate / window_size
    min_silence_duration = 0.25 * sample_rate / window_size

    result = []
    last = -1
    for k, f in enumerate(out):
        if f >= threshold:
            if last == -1:
                last = k
        elif last != -1:
            if k - last > min_speech_duration:
                result.append((last, k))
            last = -1

    if last != -1 and k - last > min_speech_duration:
        result.append((last, k))

    if not result:
        print(f"Empty for {args.wav}")
        return

    print(result)

    final = [result[0]]
    for r in result[1:]:
        f = final[-1]
        if r[0] - f[1] < min_silence_duration:
            final[-1] = (f[0], r[1])
        else:
            final.append(r)

    for f in final:
        start = f[0] * window_size / sample_rate
        end = f[1] * window_size / sample_rate
        print("{:.3f} -- {:.3f}".format(start, end))


if __name__ == "__main__":
    main()
