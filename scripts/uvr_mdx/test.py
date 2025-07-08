#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import time

import argparse
import kaldi_native_fbank as knf
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-filename",
        type=str,
        required=True,
        help="Path to onnx model",
    )

    parser.add_argument(
        "--audio-filename",
        type=str,
        required=True,
        help="Path to input audio file",
    )

    return parser.parse_args()


class OnnxModel:
    def __init__(self, filename):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 4
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts
        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        self.dim_t = self.model.get_outputs()[0].shape[3]

        self.dim_f = self.model.get_outputs()[0].shape[2]

        self.n_fft = self.dim_f * 2

        self.dim_c = self.model.get_outputs()[0].shape[1]
        assert self.dim_c == 4, self.dim_c

        self.hop = 1024
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = self.hop * (self.dim_t - 1)

        self.freq_pad = np.zeros([1, self.dim_c, self.n_bins - self.dim_f, self.dim_t])

        print(f"----------inputs for {filename}----------")
        for i in self.model.get_inputs():
            print(i)

        print(f"----------outputs for {filename}----------")

        for i in self.model.get_outputs():
            print(i)
            print(i.shape)
        print("--------------------")

    def __call__(self, x):
        """
        Args:
          x: (batch_size, 4, self.dim_f, self.dim_t)
        Returns:
          spec: (batch_size, 4, self.dim_f, self.dim_t)
        """
        spec = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x,
            },
        )[0]

        return spec


def main():
    args = get_args()
    m = OnnxModel(args.model_filename)

    stft_config = knf.StftConfig(
        n_fft=m.n_fft,
        hop_length=m.hop,
        win_length=m.n_fft,
        center=True,
        window_type="hann",
    )
    knf_stft = knf.Stft(stft_config)
    knf_istft = knf.IStft(stft_config)

    sample_rate = 44100

    samples, rate = librosa.load(args.audio_filename, mono=False, sr=sample_rate)

    start_time = time.time()

    assert rate == sample_rate, (rate, sample_rate)

    # samples: (2, 479832) , (num_channels, num_samples), 44100, 10.88
    print("samples", samples.shape, rate, samples.shape[1] / rate)

    assert samples.ndim == 2, samples.shape
    assert samples.shape[0] == 2, samples.shape

    margin = sample_rate

    num_chunks = 15
    chunk_size = num_chunks * sample_rate

    # if they are too few samples, reset chunk_size
    if samples.shape[1] < chunk_size:
        chunk_size = samples.shape[1]

    if margin > chunk_size:
        margin = chunk_size

    segments = []
    for skip in range(0, samples.shape[1], chunk_size):
        start = max(0, skip - margin)
        end = min(skip + chunk_size + margin, samples.shape[1])
        segments.append(samples[:, start:end])
        if end == samples.shape[1]:
            break

    sources = []
    for kk, s in enumerate(segments):
        num_samples = s.shape[1]
        trim = m.n_fft // 2
        gen_size = m.chunk_size - 2 * trim
        pad = gen_size - s.shape[1] % gen_size
        mix_p = np.concatenate(
            (
                np.zeros((2, trim)),
                s,
                np.zeros((2, pad)),
                np.zeros((2, trim)),
            ),
            axis=1,
        )

        chunk_list = []
        i = 0
        while i < s.shape[1] + pad:
            chunk_list.append(mix_p[:, i : i + m.chunk_size])
            i += gen_size

        mix_waves = np.array(chunk_list)

        mix_waves_reshaped = mix_waves.reshape(-1, m.chunk_size)
        stft_results = []
        for w in mix_waves_reshaped:
            stft = knf_stft(w)
            stft_results.append(stft)
        real = np.array(
            [np.array(s.real).reshape(s.num_frames, -1) for s in stft_results],
            dtype=np.float32,
        )[:, :, :-1]
        # real: (6, 256, 3072)

        real = real.transpose(0, 2, 1)
        # real: (6, 3072, 256)

        imag = np.array(
            [np.array(s.imag).reshape(s.num_frames, -1) for s in stft_results],
            dtype=np.float32,
        )[:, :, :-1]
        imag = imag.transpose(0, 2, 1)
        # imag: (6, 3072, 256)

        x = np.stack([real, imag], axis=1)
        # x: (6, 2, 3072, 256) -> (batch_size, real_imag, 3072, 256)
        x = x.reshape(-1, m.dim_c, m.dim_f, m.dim_t)
        # x: (3, 4, 3072, 256)
        spec = m(x)

        freq_pad = np.repeat(m.freq_pad, spec.shape[0], axis=0)

        x = np.concatenate([spec, freq_pad], axis=2)
        # x: (3, 4, 3073, 256)
        x = x.reshape(-1, 2, m.n_bins, m.dim_t)
        # x: (6, 2, 3073, 256)
        x = x.transpose(0, 1, 3, 2)
        # x: (6, 2, 256, 3073)
        num_frames = x.shape[2]

        x = x.reshape(x.shape[0], x.shape[1], -1)
        wav_list = []
        for k in range(x.shape[0]):
            istft_result = knf.StftResult(
                real=x[k, 0].reshape(-1).tolist(),
                imag=x[k, 1].reshape(-1).tolist(),
                num_frames=num_frames,
            )
            wav = knf_istft(istft_result)
            wav_list.append(wav)
        wav = np.array(wav_list, dtype=np.float32)
        # wav: (6, 261120)

        wav = wav.reshape(-1, 2, wav.shape[-1])
        # wav: (3, 2, 261120)

        wav = wav[:, :, trim:-trim]
        # wav: (3, 2, 254976)

        wav = wav.transpose(1, 0, 2)
        # wav: (2, 3, 254976)

        wav = wav.reshape(2, -1)
        # wav: (2, 764928)

        wav = wav[:, :-pad]
        # wav: 2, 705600)
        if kk == 0:
            start = 0
        else:
            start = margin

        if kk == len(segments) - 1:
            end = None
        else:
            end = -margin

        sources.append(wav[:, start:end])

    sources = np.concatenate(sources, axis=-1)

    vocals = sources
    non_vocals = samples - vocals
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")

    audio_duration = samples.shape[1] / sample_rate
    real_time_factor = elapsed_seconds / audio_duration
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")

    sf.write(f"./vocals.mp3", np.transpose(vocals), sample_rate)
    sf.write(f"./non_vocals.mp3", np.transpose(non_vocals), sample_rate)


if __name__ == "__main__":
    main()
