#!/usr/bin/env python3

"""
./export-onnx.py
./preprocess.sh

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
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
from pyannote.audio import Model


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
        self.receptive_field_size = int(meta["receptive_field_size"])
        self.receptive_field_shift = int(meta["receptive_field_shift"])

        self.gt = Model.from_pretrained('./pytorch_model.bin')
        self.gt.eval()

    def __call__(self, x):
        """
        Args:
          x: (N, num_samples)
        Returns:
          A tensor of shape (N, num_frames, num_classes)
        """
        x = np.expand_dims(x, axis=1)
        return self.gt(torch.from_numpy(x)).numpy()

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
    Returns:
      A tensor of shape (num_chunks, num_frames, num_speakers)
    """
    y = np.argmax(y, axis=-1)
    mapping = get_powerset_mapping(7, 3, 2)
    labels = mapping[y.reshape(-1)].reshape(y.shape[0], y.shape[1], -1)
    return labels


@torch.no_grad()
def main():
    args = get_args()
    assert Path(args.model).is_file(), args.model
    assert Path(args.wav).is_file(), args.wav

    m = OnnxModel(args.model)
    audio = load_wav(args.wav, m.sample_rate)
    # audio: (num_samples,)
    print(audio.shape, audio.min(), audio.max(), audio.sum())

    num = (audio.shape[0] - m.window_size) // m.window_shift + 1

    samples = as_strided(
        audio,
        shape=(num, m.window_size),
        strides=(m.window_shift * audio.strides[0], audio.strides[0]),
    )

    # or use torch.Tensor.unfold
    samples = torch.from_numpy(audio).unfold(0, m.window_size, m.window_shift).numpy()

    print(
        "samples",
        samples.shape,
        samples.mean(),
        samples.reshape(-1).sum(),
        samples[:3, :3].sum(axis=-1),
    )

    if (
        audio.shape[0] < m.window_size
        or (audio.shape[0] - m.window_size) % m.window_shift > 0
    ):
        has_last_chunk = True
    else:
        has_last_chunk = False

    num_chunks = samples.shape[0]
    batch_size = 32
    output = []
    for i in range(0, num_chunks, batch_size):
        start = i
        end = i + batch_size
        # it's perfectly ok to use end > num_chunks
        print(
            "here samples",
            samples[start:end].shape,
            samples[start:end].sum(),
            samples[start:end].mean(),
        )
        y = m(samples[start:end])
        print("here y", y.shape, y.sum(), y.mean())

        k = to_multi_label(y)
        print("here k", k.shape, k.sum(), k.mean())
        output.append(y)

    if has_last_chunk:
        last_chunk = audio[num_chunks * m.window_shift :]
        pad_size = m.window_size - last_chunk.shape[0]
        print('last samples', last_chunk.shape, last_chunk.sum(), last_chunk.mean())
        last_chunk = np.pad(last_chunk, (0, pad_size))
        last_chunk = np.expand_dims(last_chunk, axis=0)
        y = m(last_chunk)
        print('last', y.shape, y.sum(), y.mean())
        output.append(y)

    y = np.vstack(output)

    print("y", y.sum(), y.mean(), y.shape)
    labels = to_multi_label(y)
    # labels: (num_chunks, num_frames, num_speakers)
    print("multi", labels.sum(), labels.mean())

    # binary classification
    labels = np.max(labels, axis=-1)
    # labels: (num_chunk, num_frames)
    print("labels.shape", labels.shape, labels.sum(), labels.mean())

    num_frames = (
        int(
            (m.window_size + (labels.shape[0] - 1) * m.window_shift)
            / m.receptive_field_shift
        )
        + 1
    )
    print("num_frames", num_frames)

    count = np.zeros((num_frames,))
    classification = np.zeros((num_frames,))
    ones = np.ones((labels.shape[1],))
    weight = np.hamming(labels.shape[1])
    #  weight = np.ones(labels.shape[1])
    print('weight', weight.shape, weight.sum(), weight.mean())

    for i in range(labels.shape[0]):
        this_chunk = labels[i]
        start = int(i * m.window_shift  / m.receptive_field_shift + 0.5)
        end = start + this_chunk.shape[0]

        classification[start:end] += this_chunk * weight
        count[start:end] += weight

    print('classification', classification.shape, classification.sum(), classification.mean())
    print('count', count.shape, count.sum(), count.mean())
    classification /= np.maximum(count, 1e-12)
    print('average', classification.shape, classification.sum(), classification.mean())

    if has_last_chunk:
        stop_frame = int(audio.shape[0] / m.receptive_field_shift)
        classification = classification[:stop_frame]
        print('stop_frame', stop_frame)
    print('final', classification.shape, classification.sum(), classification.mean())

    classification = classification.tolist()

    onset = 0.5
    offset = 0.5

    is_active = classification[0] > onset
    start = None

    scale = m.receptive_field_shift / m.sample_rate
    scale_offset = m.receptive_field_size / m.sample_rate * 0.5
    print(scale, offset)

    for i in range(len(classification)):
        #  print(i, classification[i])
        if is_active:
            if classification[i] < offset:
                #  print('on->off', start , start*scale+scale_offset, i, classification[i])
                print(f"{start*scale + scale_offset:.3f} -- {i*scale + scale_offset:.3f}")
                is_active = False
        else:
            if classification[i] > onset:
                start = i
                #  print('off->on', start, start*scale+scale_offset, classification[i])
                is_active = True

    if is_active:
        print('last')
        print(
            f"{start*scale + scale_offset:.3f} -- {(len(classification)-1)*scale + scale_offset:.3f}"
        )


if __name__ == "__main__":
    main()
