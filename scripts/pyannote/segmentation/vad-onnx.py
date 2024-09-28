#!/usr/bin/env python3

"""
./export-onnx.py
./preprocess.sh

./vad-onnx.py --model ./model.onnx --wav ./lei-jun-test.wav
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from numpy.lib.stride_tricks import as_strided


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model.onnx")
    parser.add_argument("--wav", type=str, required=True, help="Path to test.wav")

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
        print(meta)

        self.window_size = int(meta["window_size"])
        self.sample_rate = int(meta["sample_rate"])
        self.window_shift = int(0.1 * self.window_size)

    def __call__(self, x):
        """
        Args:
          x: (N, num_samples)
        Returns:
          A tensor of shape (N, num_frames, num_classes)
        """
        x = np.expand_dims(x, axis=1)
        (y,) = self.model.run(
            [self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: x}
        )

        return y


def load_wav(filename, expected_sample_rate) -> np.ndarray:
    audio, sample_rate = sf.read(filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel
    if sample_rate != expected_sample_rate:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=expected_sample_rate,
        )
    return audio


def get_powerset_mapping(num_rows, num_speakers, max_set_size):
    mapping = np.zeros((num_rows, num_speakers))

    k = 1
    for i in range(1, max_set_size + 1):
        if i == 1:
            for j in range(0, num_speakers):
                mapping[k, j] = 1
                k += 1
        elif i == 2:
            for j in range(0, num_speakers):
                for m in range(j + 1, num_speakers):
                    mapping[k, j] = 1
                    mapping[k, m] = 1
                    k += 1
        elif i == 3:
            raise RuntimeError("Unsupported")

    return mapping


def to_multi_label(y):
    """
    Args:
      y: (num_chunks, num_frames, num_classes)
    """
    y = np.argmax(y, axis=-1)
    mapping = get_powerset_mapping(7, 3, 2)
    labels = mapping[y.reshape(-1)].reshape(y.shape[0], y.shape[1], -1)
    return labels


def main():
    args = get_args()
    assert Path(args.model).is_file(), args.model
    assert Path(args.wav).is_file(), args.wav

    m = OnnxModel(args.model)
    audio = load_wav(args.wav, m.sample_rate)
    # audio: (num_samples,)
    print(audio.shape, audio.min(), audio.max())

    num = (audio.shape[0] - m.window_size) // m.window_shift + 1

    samples = as_strided(
        audio,
        shape=(num, m.window_size),
        strides=(m.window_shift * audio.strides[0], audio.strides[0]),
    )
    # TODO(fangjun): Pad the last chunk if any
    # samples: (num_chunks, window_size)

    # or use torch.Tensor.unfold
    #  samples = torch.from_numpy(audio).unfold(0, m.window_size, m.window_shift).numpy()

    num_chunks = samples.shape[0]
    batch_size = 3
    output = []
    for i in range(0, num_chunks, batch_size):
        start = i
        end = i + batch_size
        # it's perfectly ok to use end > num_chunks
        y = m(samples[start:end])
        output.append(y)
    y = np.vstack(output)
    labels = to_multi_label(y)

    # binary classification
    labels = np.max(labels, axis=-1)
    # labels: (num_chunk, num_frames)

    num_frames = int(audio.shape[0] / m.window_size * labels.shape[1]) + 1

    count = np.zeros((num_frames,))
    classification = np.zeros((num_frames,))
    ones = np.ones((labels.shape[1],))

    for i in range(labels.shape[0]):
        this_chunk = labels[i]
        start = int(i * m.window_shift / m.window_size * labels.shape[1])
        end = start + this_chunk.shape[0]

        classification[start:end] += this_chunk
        count[start:end] += ones

    classification /= count + 1e-5

    classification = classification.tolist()

    is_active = classification[0] > 0.5
    start = None

    scale = 10 / labels.shape[1]

    for i in range(len(classification)):
        if is_active:
            if classification[i] < 0.5:
                print(f"{start*scale:.2f} -- {i*scale: .2f}")
                is_active = False
        else:
            if classification[i] > 0.5:
                start = i
                is_active = True


if __name__ == "__main__":
    main()
