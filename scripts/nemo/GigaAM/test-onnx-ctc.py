#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

# https://github.com/salute-developers/GigaAM

import kaldi_native_fbank as knf
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch


def create_fbank():
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.remove_dc_offset = False
    opts.frame_opts.preemph_coeff = 0
    opts.frame_opts.window_type = "hann"

    # Even though GigaAM uses 400 for fft, here we use 512
    # since kaldi-native-fbank only support fft for power of 2.
    opts.frame_opts.round_to_power_of_two = True

    opts.mel_opts.low_freq = 0
    opts.mel_opts.high_freq = 8000
    opts.mel_opts.num_bins = 64

    fbank = knf.OnlineFbank(opts)
    return fbank


def compute_features(audio, fbank) -> np.ndarray:
    """
    Args:
      audio: (num_samples,), np.float32
      fbank: the fbank extractor
    Returns:
      features: (num_frames, feat_dim), np.float32
    """
    assert len(audio.shape) == 1, audio.shape
    fbank.accept_waveform(16000, audio)
    ans = []
    processed = 0
    while processed < fbank.num_frames_ready:
        ans.append(np.array(fbank.get_frame(processed)))
        processed += 1
    ans = np.stack(ans)
    return ans


def display(sess):
    print("==========Input==========")
    for i in sess.get_inputs():
        print(i)
    print("==========Output==========")
    for i in sess.get_outputs():
        print(i)


"""
==========Input==========
NodeArg(name='audio_signal', type='tensor(float)', shape=['audio_signal_dynamic_axes_1', 64, 'audio_signal_dynamic_axes_2'])
NodeArg(name='length', type='tensor(int64)', shape=['length_dynamic_axes_1'])
==========Output==========
NodeArg(name='logprobs', type='tensor(float)', shape=['logprobs_dynamic_axes_1', 'logprobs_dynamic_axes_2', 34])
"""


class OnnxModel:
    def __init__(
        self,
        filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.model = ort.InferenceSession(
            filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        display(self.model)

    def __call__(self, x: np.ndarray):
        # x: (T, C)
        x = torch.from_numpy(x)
        x = x.t().unsqueeze(0)
        # x: [1, C, T]
        x_lens = torch.tensor([x.shape[-1]], dtype=torch.int64)

        log_probs = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_lens.numpy(),
            },
        )[0]
        # [batch_size, T, dim]
        return log_probs


def main():
    filename = "./model.int8.onnx"
    tokens = "./tokens.txt"
    wav = "./example.wav"

    model = OnnxModel(filename)

    id2token = dict()
    with open(tokens, encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) == 1:
                id2token[int(fields[0])] = " "
            else:
                t, idx = fields
                id2token[int(idx)] = t

    fbank = create_fbank()
    audio, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel
    if sample_rate != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )
        sample_rate = 16000

    features = compute_features(audio, fbank)
    print("features.shape", features.shape)

    blank = len(id2token) - 1
    prev = -1
    ans = []
    log_probs = model(features)
    print("log_probs", log_probs.shape)
    log_probs = torch.from_numpy(log_probs)[0]
    ids = torch.argmax(log_probs, dim=1).tolist()
    for i in ids:
        if i != blank and i != prev:
            ans.append(i)
        prev = i

    tokens = [id2token[i] for i in ans]

    text = "".join(tokens)
    print(wav)
    print(text)


if __name__ == "__main__":
    main()
