#!/usr/bin/env python3
# Copyright      2023-2024  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This script computes speaker similarity score in the range [0-1]
of two wave files using a speaker embedding model.
"""
import argparse
import wave
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
from numpy.linalg import norm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the input onnx model. Example value: model.onnx",
    )

    parser.add_argument(
        "--file1",
        type=str,
        required=True,
        help="Input wave 1",
    )

    parser.add_argument(
        "--file2",
        type=str,
        required=True,
        help="Input wave 2",
    )

    return parser.parse_args()


def read_wavefile(filename, expected_sample_rate: int = 16000) -> np.ndarray:
    """
    Args:
      filename:
        Path to a wave file, which must be of 16-bit and 16kHz.
     expected_sample_rate:
       Expected sample rate of the wave file.
    Returns:
      Return a 1-D float32 array containing audio samples. Each sample is in
      the range [-1, 1].
    """
    filename = str(filename)
    with wave.open(filename) as f:
        wave_file_sample_rate = f.getframerate()
        assert wave_file_sample_rate == expected_sample_rate, (
            wave_file_sample_rate,
            expected_sample_rate,
        )

        num_channels = f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_int16 = samples_int16.reshape(-1, num_channels)[:, 0]
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768

        return samples_float32


def compute_features(samples: np.ndarray, model: "OnnxModel") -> np.ndarray:
    fbank_opts = knf.FbankOptions()
    fbank_opts.frame_opts.samp_freq = model.sample_rate
    fbank_opts.frame_opts.frame_length_ms = model.window_size_ms
    fbank_opts.frame_opts.frame_shift_ms = model.window_stride_ms
    fbank_opts.frame_opts.dither = 0
    fbank_opts.frame_opts.remove_dc_offset = False
    fbank_opts.frame_opts.window_type = model.window_type

    fbank_opts.mel_opts.num_bins = model.feat_dim
    fbank_opts.mel_opts.low_freq = 0
    fbank_opts.mel_opts.is_librosa = True

    fbank = knf.OnlineFbank(fbank_opts)
    fbank.accept_waveform(model.sample_rate, samples)
    fbank.input_finished()

    features = []
    for i in range(fbank.num_frames_ready):
        f = fbank.get_frame(i)
        features.append(f)
    features = np.stack(features, axis=0)
    # at this point, the shape of features is (T, C)

    if model.feature_normalize_type != "":
        assert model.feature_normalize_type == "per_feature"
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        features = (features - mean) / std

    feature_len = features.shape[0]
    pad = 16 - feature_len % 16

    if pad > 0:
        padding = np.zeros((pad, features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, padding])

    features = np.expand_dims(features, axis=0)

    return features, feature_len


class OnnxModel:
    def __init__(
        self,
        filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
        )

        meta = self.model.get_modelmeta().custom_metadata_map
        self.framework = meta["framework"]
        self.sample_rate = int(meta["sample_rate"])
        self.output_dim = int(meta["output_dim"])
        self.feature_normalize_type = meta["feature_normalize_type"]
        self.window_size_ms = int(meta["window_size_ms"])
        self.window_stride_ms = int(meta["window_stride_ms"])
        self.window_type = meta["window_type"]
        self.feat_dim = int(meta["feat_dim"])
        print(meta)

        assert self.framework == "nemo", self.framework

    def __call__(self, x: np.ndarray, x_lens: int) -> np.ndarray:
        """
        Args:
          x:
            A 2-D float32 tensor of shape (T, C).
          y:
            A 1-D float32 tensor containing model output.
        """
        x = x.transpose(0, 2, 1)  # (B, T, C) -> (B, C, T)
        x_lens = np.asarray([x_lens], dtype=np.int64)

        return self.model.run(
            [
                self.model.get_outputs()[1].name,
            ],
            {
                self.model.get_inputs()[0].name: x,
                self.model.get_inputs()[1].name: x_lens,
            },
        )[0][0]


def main():
    args = get_args()
    print(args)
    filename = Path(args.model)
    file1 = Path(args.file1)
    file2 = Path(args.file2)
    assert filename.is_file(), filename
    assert file1.is_file(), file1
    assert file2.is_file(), file2

    model = OnnxModel(filename)
    wave1 = read_wavefile(file1, model.sample_rate)
    wave2 = read_wavefile(file2, model.sample_rate)

    features1, features1_len = compute_features(wave1, model)
    features2, features2_len = compute_features(wave2, model)

    output1 = model(features1, features1_len)
    output2 = model(features2, features2_len)

    similarity = np.dot(output1, output2) / (norm(output1) * norm(output2))
    print(f"similarity in the range [0-1]: {similarity}")


if __name__ == "__main__":
    main()
