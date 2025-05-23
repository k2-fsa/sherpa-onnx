#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
import time

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from separate import load_audio

"""
----------inputs for ./2stems/vocals.onnx----------
NodeArg(name='x', type='tensor(float)', shape=['num_splits', 2, 512, 1024])
----------outputs for ./2stems/vocals.onnx----------
NodeArg(name='y', type='tensor(float)', shape=['Muly_dim_0', 2, 512, 1024])

----------inputs for ./2stems/accompaniment.onnx----------
NodeArg(name='x', type='tensor(float)', shape=['num_splits', 2, 512, 1024])
----------outputs for ./2stems/accompaniment.onnx----------
NodeArg(name='y', type='tensor(float)', shape=['Muly_dim_0', 2, 512, 1024])

"""


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

        print(f"----------inputs for {filename}----------")
        for i in self.model.get_inputs():
            print(i)

        print(f"----------outputs for {filename}----------")

        for i in self.model.get_outputs():
            print(i)
        print("--------------------")

    def __call__(self, x):
        """
        Args:
          x: (num_splits, 2, 512, 1024)
        """
        spec = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
            },
        )[0]

        return torch.from_numpy(spec)


def main():
    vocals = OnnxModel("./2stems/vocals.onnx")
    accompaniment = OnnxModel("./2stems/accompaniment.onnx")

    waveform, sample_rate = load_audio("./qi-feng-le.mp3")
    waveform = waveform[: 44100 * 10, :]

    stft_config = knf.StftConfig(
        n_fft=4096,
        hop_length=1024,
        win_length=4096,
        center=False,
        window_type="hann",
    )
    knf_stft = knf.Stft(stft_config)
    knf_istft = knf.IStft(stft_config)

    start = time.time()

    stft_result_c0 = knf_stft(waveform[:, 0].tolist())
    stft_result_c1 = knf_stft(waveform[:, 1].tolist())
    print("c0 stft", stft_result_c0.num_frames)

    orig_real0 = np.array(stft_result_c0.real, dtype=np.float32).reshape(
        stft_result_c0.num_frames, -1
    )
    orig_imag0 = np.array(stft_result_c0.imag, dtype=np.float32).reshape(
        stft_result_c0.num_frames, -1
    )

    orig_real1 = np.array(stft_result_c1.real, dtype=np.float32).reshape(
        stft_result_c1.num_frames, -1
    )
    orig_imag1 = np.array(stft_result_c1.imag, dtype=np.float32).reshape(
        stft_result_c1.num_frames, -1
    )

    real0 = torch.from_numpy(orig_real0)
    imag0 = torch.from_numpy(orig_imag0)
    real1 = torch.from_numpy(orig_real1)
    imag1 = torch.from_numpy(orig_imag1)
    # (num_frames, n_fft/2_1)
    print("real0", real0.shape)

    # keep only the first 1024 bins
    real0 = real0[:, :1024]
    imag0 = imag0[:, :1024]
    real1 = real1[:, :1024]
    imag1 = imag1[:, :1024]

    stft0 = (real0.square() + imag0.square()).sqrt()
    stft1 = (real1.square() + imag1.square()).sqrt()

    # pad it to multiple of 512
    padding = 512 - real0.shape[0] % 512
    print("padding", padding)
    if padding > 0:
        stft0 = torch.nn.functional.pad(stft0, (0, 0, 0, padding))
        stft1 = torch.nn.functional.pad(stft1, (0, 0, 0, padding))
    stft0 = stft0.reshape(-1, 1, 512, 1024)
    stft1 = stft1.reshape(-1, 1, 512, 1024)

    stft_01 = torch.cat([stft0, stft1], axis=1)

    print("stft_01", stft_01.shape, stft_01.dtype)

    vocals_spec = vocals(stft_01)
    accompaniment_spec = accompaniment(stft_01)
    # (num_splits, num_channels, 512, 1024)

    sum_spec = (vocals_spec.square() + accompaniment_spec.square()) + 1e-10

    vocals_spec = (vocals_spec**2 + 1e-10 / 2) / sum_spec
    accompaniment_spec = (accompaniment_spec**2 + 1e-10 / 2) / sum_spec

    for name, spec in zip(
        ["vocals", "accompaniment"], [vocals_spec, accompaniment_spec]
    ):
        spec_c0 = spec[:, 0, :, :]
        spec_c1 = spec[:, 1, :, :]

        spec_c0 = spec_c0.reshape(-1, 1024)
        spec_c1 = spec_c1.reshape(-1, 1024)

        spec_c0 = spec_c0[: stft_result_c0.num_frames, :]
        spec_c1 = spec_c1[: stft_result_c0.num_frames, :]

        spec_c0 = torch.nn.functional.pad(spec_c0, (0, 2049 - 1024, 0, 0))
        spec_c1 = torch.nn.functional.pad(spec_c1, (0, 2049 - 1024, 0, 0))

        spec_c0_real = spec_c0 * orig_real0
        spec_c0_imag = spec_c0 * orig_imag0

        spec_c1_real = spec_c1 * orig_real1
        spec_c1_imag = spec_c1 * orig_imag1

        result0 = knf.StftResult(
            real=spec_c0_real.reshape(-1).tolist(),
            imag=spec_c0_imag.reshape(-1).tolist(),
            num_frames=orig_real0.shape[0],
        )

        result1 = knf.StftResult(
            real=spec_c1_real.reshape(-1).tolist(),
            imag=spec_c1_imag.reshape(-1).tolist(),
            num_frames=orig_real1.shape[0],
        )

        wav0 = knf_istft(result0)
        wav1 = knf_istft(result1)

        wav = np.array([wav0, wav1], dtype=np.float32)
        wav = np.transpose(wav)
        # now wav is (num_samples, num_channels)

        sf.write(f"./onnx-{name}.wav", wav, 44100)

        print(f"Saved to ./onnx-{name}.wav")

    end = time.time()
    elapsed_seconds = end - start
    audio_duration = waveform.shape[0] / sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    main()
