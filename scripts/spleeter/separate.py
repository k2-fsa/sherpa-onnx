#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

# Please see ./run.sh for usage

from typing import Optional

import ffmpeg
import numpy as np
import torch
import soundfile as sf
import torchaudio

from unet import UNet


def load_audio(filename, sample_rate: Optional[int] = 44100):
    probe = ffmpeg.probe(filename)
    if "streams" not in probe or len(probe["streams"]) == 0:
        raise ValueError("No stream was found with ffprobe")

    metadata = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "audio"
    )
    n_channels = metadata["channels"]

    if sample_rate is None:
        sample_rate = metadata["sample_rate"]

    process = (
        ffmpeg.input(filename)
        .output("pipe:", format="f32le", ar=sample_rate)
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    buffer, _ = process.communicate()
    waveform = np.frombuffer(buffer, dtype="<f4").reshape(-1, n_channels)
    waveform = torch.from_numpy(waveform).to(torch.float32)
    return waveform, sample_rate


@torch.no_grad()
def main():
    vocals = UNet()
    vocals.eval()
    state_dict = torch.load("./2stems/vocals.pt", map_location="cpu")
    vocals.load_state_dict(state_dict)

    accompaniment = UNet()
    accompaniment.eval()
    state_dict = torch.load("./2stems/accompaniment.pt", map_location="cpu")
    accompaniment.load_state_dict(state_dict)

    waveform, sample_rate = load_audio("./audio_example.mp3")
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
    print("stft", stft.shape)

    # stft: (2, 2049, 465)
    # stft is a complex tensor
    y = stft.permute(2, 1, 0)
    print("y0", y.shape)
    # (465, 2049, 2)

    y = y[:, :1024, :]
    # (465, 1024, 2)

    tensor_size = y.shape[0] - int(y.shape[0] / 512) * 512
    pad_size = 512 - tensor_size
    y = torch.nn.functional.pad(y, (0, 0, 0, 0, 0, pad_size))
    # (512, 1024, 2)
    print("y1", y.shape, y.dtype)

    num_splits = int(y.shape[0] / 512)
    y = y.reshape([num_splits, 512] + list(y.shape[1:]))
    # y: (1, 512, 1024, 2)
    print("y2", y.shape, y.dtype)

    y = y.abs()
    y = y.permute(0, 3, 1, 2)
    # (1, 2, 512, 1024)
    print("y3", y.shape, y.dtype)

    vocals_out = vocals(y)
    accompaniment_out = accompaniment(y)

    sum_out = (vocals_out**2 + accompaniment_out**2) + 1e-10
    print("vocals_out", vocals_out.shape, accompaniment_out.shape, sum_out.shape)

    vocals_mask = (vocals_out**2 + 1e-10 / 2) / sum_out
    # (1, 2, 512, 1024)

    vocals_mask = torch.nn.functional.pad(
        vocals_mask, (0, 2049 - 1024, 0, 0, 0, 0, 0, 0)
    )
    # (1, 2, 512, 2049)

    vocals_mask = vocals_mask.permute(0, 2, 3, 1)
    # (1, 512, 2049, 2)
    print("here00", vocals_mask.shape)

    vocals_mask = vocals_mask.reshape(-1, vocals_mask.shape[2], vocals_mask.shape[3])
    # (512, 2049, 2)

    print("here2", vocals_mask.shape)
    # (512, 2049, 2)

    vocals_mask = vocals_mask[: stft.shape[2], :, :]
    # (465, 2049, 2)
    print("here 3", vocals_mask.shape, stft.shape)

    vocals_mask = vocals_mask.permute(2, 1, 0)
    # (2, 2049, 465)
    print(vocals_mask.sum())
    print(vocals_mask.mean())

    masked_vocals_stft = vocals_mask * stft

    print("final", vocals_mask.shape, stft.shape, masked_vocals_stft.shape)

    wave = torch.istft(
        masked_vocals_stft,
        4096,
        1024,
        window=torch.hann_window(4096, periodic=True),
        onesided=True,
    ) * (2 / 3)
    print(wave.shape, wave.dtype)
    sf.write("a.wav", wave.t(), 44100)


if __name__ == "__main__":
    main()
