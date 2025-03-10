#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


class OnnxModel:
    def __init__(self):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts
        self.model = ort.InferenceSession(
            "./gtcrn_simple.onnx",
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        meta = self.model.get_modelmeta().custom_metadata_map
        self.sample_rate = int(meta["sample_rate"])
        self.n_fft = int(meta["n_fft"])
        self.hop_length = int(meta["hop_length"])
        self.window_length = int(meta["window_length"])
        assert meta["window_type"] == "hann_sqrt", meta["window_type"]

        self.window = torch.hann_window(self.window_length).pow(0.5)

    def get_init_states(self):
        meta = self.model.get_modelmeta().custom_metadata_map
        conv_cache_shape = list(map(int, meta["conv_cache_shape"].split(",")))
        tra_cache_shape = list(map(int, meta["tra_cache_shape"].split(",")))
        inter_cache_shape = list(map(int, meta["inter_cache_shape"].split(",")))

        conv_cache_shape = np.zeros(conv_cache_shape, dtype=np.float32)
        tra_cache = np.zeros(tra_cache_shape, dtype=np.float32)
        inter_cache = np.zeros(inter_cache_shape, dtype=np.float32)

        return conv_cache_shape, tra_cache, inter_cache

    def __call__(self, x, states):
        """
        Args:
          x: (1, n_fft/2+1, 1, 2)
        Returns:
          o: (1, n_fft/2+1, 1, 2)
        """
        out, next_conv_cache, next_tra_cache, next_inter_cache = self.model.run(
            [
                self.model.get_outputs()[0].name,
                self.model.get_outputs()[1].name,
                self.model.get_outputs()[2].name,
                self.model.get_outputs()[3].name,
            ],
            {
                self.model.get_inputs()[0].name: x,
                self.model.get_inputs()[1].name: states[0],
                self.model.get_inputs()[2].name: states[1],
                self.model.get_inputs()[3].name: states[2],
            },
        )

        return out, (next_conv_cache, next_tra_cache, next_inter_cache)


def main():
    model = OnnxModel()

    filename = "./inp_16k.wav"
    wave, sample_rate = load_audio(filename)
    if sample_rate != model.sample_rate:
        import librosa

        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=model.sample_rate)
        sample_rate = model.sample_rate

    stft_config = knf.StftConfig(
        n_fft=model.n_fft,
        hop_length=model.hop_length,
        win_length=model.window_length,
        window=model.window.tolist(),
    )
    stft = knf.Stft(stft_config)
    stft_result = stft(wave)
    num_frames = stft_result.num_frames
    real = np.array(stft_result.real, dtype=np.float32).reshape(num_frames, -1)
    imag = np.array(stft_result.imag, dtype=np.float32).reshape(num_frames, -1)

    states = model.get_init_states()
    outputs = []
    for i in range(num_frames):
        x_real = real[i : i + 1]
        x_imag = imag[i : i + 1]
        x = np.vstack([x_real, x_imag]).transpose()
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=2)

        o, states = model(x, states)
        outputs.append(o)

    outputs = np.concatenate(outputs, axis=2)
    outputs = outputs.squeeze(0).transpose(1, 0, 2)

    enhanced_real = outputs[:, :, 0]
    enhanced_imag = outputs[:, :, 1]
    enhanced_stft_result = knf.StftResult(
        real=enhanced_real.reshape(-1).tolist(),
        imag=enhanced_imag.reshape(-1).tolist(),
        num_frames=enhanced_real.shape[0],
    )

    istft = knf.IStft(stft_config)
    enhanced = istft(enhanced_stft_result)

    sf.write("./enhanced_16k.wav", enhanced, model.sample_rate)


if __name__ == "__main__":
    main()
