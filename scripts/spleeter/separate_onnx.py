#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
import torch
import onnxruntime as ort
from separate import load_audio
import soundfile as sf

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
    print("waveform", waveform.shape, sample_rate)
    # waveform torch.Size([13760640, 2]) 44100

    # Support only 2 channels at present. If you have only a single channel,
    # please replicate it to get 2 channels
    assert waveform.shape[1] == 2, waveform.shape

    # torch.stft requires a 2-D input of shape (N, T), so we transpose waveform
    stft = torch.stft(
        waveform.t(),
        n_fft=4096,
        hop_length=1024,
        window=torch.hann_window(4096, periodic=True),
        center=False,
        onesided=True,
        return_complex=True,
    )
    print("stft", stft.shape, stft.dtype)
    # stft: (2, 2049, 13435), torch.complex64
    # (num_channels, n_fft/2+1, num_frames)

    y = stft.permute(2, 1, 0)
    print("y0", y.shape)
    # y: (13435, 2049, 2) -> (num_frames, n_fft/2+1, num_channels)

    y = y[:, :1024, :]
    print("y1", y.shape)
    # y1: (13435, 1024, 2) -> (num_frames, 1024, num_channels)

    tensor_size = y.shape[0] - int(y.shape[0] / 512) * 512
    pad_size = 512 - tensor_size
    y = torch.nn.functional.pad(y, (0, 0, 0, 0, 0, pad_size))
    print("y2", y.shape, y.dtype)
    # y2: (13824, 1024, 2) -> (num_frames, 1024, num_channels)
    num_splits = y.shape[0] // 512
    y = y.reshape([num_splits, 512] + list(y.shape[1:]))
    print("y3", y.shape)
    # y3: (27, 512, 1024, 2) -> (num_splits, 512, 1024, num_channels)

    y = y.abs()
    print("y4", y.shape, y.dtype)
    # y4: (27, 512, 1024, 2), torch.float32

    y = y.permute(0, 3, 1, 2)
    print("y5", y.shape)
    # y5: (27, 2, 512, 1024) -> (num_splits, num_channels, 512, 1024)

    vocals_spec = vocals(y)
    print("vocals_spec1", vocals_spec.shape, vocals_spec.dtype)
    # vocals_spec1: (27, 2, 512, 1024)

    accompaniment_spec = accompaniment(y)
    print("accompaniment_spec1", accompaniment_spec.shape, accompaniment_spec.dtype)
    # accompaniment_spec1: (27, 2, 512, 1024)

    sum_spec = (vocals_spec**2 + accompaniment_spec**2) + 1e-10

    vocals_spec = (vocals_spec**2 + 1e-10 / 2) / sum_spec
    accompaniment_spec = (accompaniment_spec**2 + 1e-10 / 2) / sum_spec

    print("vocals_spec2", vocals_spec.shape, vocals_spec.dtype)
    # (27, 2, 512, 1024)
    print("accompaniment_spec2", accompaniment_spec.shape, accompaniment_spec.dtype)
    # (27, 2, 512, 1024)

    for name, spec in zip(
        ["vocals", "accompaniment"], [vocals_spec, accompaniment_spec]
    ):
        spec = torch.nn.functional.pad(spec, (0, 2049 - 1024, 0, 0, 0, 0, 0, 0))
        print("spec.shape", spec.shape)
        # (27, 2, 512, 2049)

        spec = spec.permute(0, 2, 3, 1)
        # (27, 512, 2049, 2)

        spec = spec.reshape(-1, spec.shape[2], spec.shape[3])
        # (512, 2049, 2)

        print("here2", spec.shape)
        # (13824, 2049, 2)

        spec = spec[: stft.shape[2], :, :]
        # (13435, 2049, 2)

        spec = spec.permute(2, 1, 0)
        print("spec.dtype", spec.shape, spec.dtype)
        # (2, 2049, 465)

        masked_stft = spec * stft
        print("masked_stft", masked_stft.shape, masked_stft.dtype)

        wave = torch.istft(
            masked_stft,
            4096,
            1024,
            window=torch.hann_window(4096, periodic=True),
            onesided=True,
        ) * (2 / 3)

        print(wave.shape, wave.dtype)
        sf.write(f"{name}.wav", wave.t(), 44100)


if __name__ == "__main__":
    main()
