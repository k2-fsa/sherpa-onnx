#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import datetime as dt

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )


class OnnxVocosModel:
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
            providers=["CPUExecutionProvider"],
        )

        print("----------vocos----------")
        for i in self.model.get_inputs():
            print(i)

        print("-----")

        for i in self.model.get_outputs():
            print(i)
        print()

    def __call__(self, x: np.ndarray):
        """
        Args:
          x: (N, feat_dim, num_frames)
        Returns:
          mag: (N, n_fft/2+1, num_frames)
          x: (N, n_fft/2+1, num_frames)
          y: (N, n_fft/2+1, num_frames)

        The complex spectrum is mag * (x + j*y)
        """
        assert x.ndim == 3, x.shape
        assert x.shape[0] == 1, x.shape

        mag, x, y = self.model.run(
            [
                self.model.get_outputs()[0].name,
                self.model.get_outputs()[1].name,
                self.model.get_outputs()[2].name,
            ],
            {
                self.model.get_inputs()[0].name: x,
            },
        )

        return mag, x, y


class OnnxHifiGANModel:
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
            providers=["CPUExecutionProvider"],
        )

        print("----------hifigan----------")
        for i in self.model.get_inputs():
            print(i)

        print("-----")

        for i in self.model.get_outputs():
            print(i)
        print()

    def __call__(self, x: np.ndarray):
        """
        Args:
          x: (N, feat_dim, num_frames)
        Returns:
          audio: (N, num_samples)
        """
        assert x.ndim == 3, x.shape
        assert x.shape[0] == 1, x.shape

        audio = self.model.run(
            [self.model.get_outputs()[0].name],
            {
                self.model.get_inputs()[0].name: x,
            },
        )[0]
        # audio: (batch_size, num_samples)

        return audio


def load_tokens(filename):
    token2id = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 1:
                t = " "
                idx = int(fields[0])
            else:
                t, idx = line.strip().split()
            token2id[t] = int(idx)
    return token2id


class OnnxModel:
    def __init__(
        self,
        filename: str,
        tokens: str,
    ):
        self.token2id = load_tokens(tokens)
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts
        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        print(f"{self.model.get_modelmeta().custom_metadata_map}")
        metadata = self.model.get_modelmeta().custom_metadata_map
        self.sample_rate = int(metadata["sample_rate"])

        print("----------matcha----------")
        for i in self.model.get_inputs():
            print(i)

        print("-----")

        for i in self.model.get_outputs():
            print(i)
        print()

    def __call__(self, x: np.ndim):
        """
        Args:
        """
        assert x.ndim == 2, x.shape
        assert x.shape[0] == 1, x.shape

        x_lengths = np.array([x.shape[1]], dtype=np.int64)

        noise_scale = np.array([1.0], dtype=np.float32)
        length_scale = np.array([1.0], dtype=np.float32)

        mel = self.model.run(
            [self.model.get_outputs()[0].name],
            {
                self.model.get_inputs()[0].name: x,
                self.model.get_inputs()[1].name: x_lengths,
                self.model.get_inputs()[2].name: noise_scale,
                self.model.get_inputs()[3].name: length_scale,
            },
        )[0]
        # mel: (batch_size, feat_dim, num_frames)

        return mel


def main():
    am = OnnxModel(
        filename="./matcha-icefall-en_US-ljspeech/model-steps-3.onnx",
        tokens="./matcha-icefall-en_US-ljspeech/tokens.txt",
    )
    vocoder = OnnxHifiGANModel("./hifigan_v2.onnx")
    vocos = OnnxVocosModel("./mel_spec_22khz_univ.onnx")

    text = "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar."
    tokens_list = phonemize_espeak(text, "en-us")
    print(tokens_list)
    tokens = []
    for t in tokens_list:
        tokens.extend(t)

    token_ids = []
    for t in tokens:
        if t not in am.token2id:
            print(f"Skip OOV '{t}'")
            continue
        token_ids.append(am.token2id[t])

    token_ids2 = [am.token2id["_"]] * (len(token_ids) * 2 + 1)
    token_ids2[1::2] = token_ids
    token_ids = token_ids2
    x = np.array([token_ids], dtype=np.int64)

    mel_start_t = dt.datetime.now()
    mel = am(x)
    mel_end_t = dt.datetime.now()

    print("mel", mel.shape)
    # mel:(1, 80, 78)

    vocos_start_t = dt.datetime.now()
    mag, x, y = vocos(mel)
    stft_result = knf.StftResult(
        real=(mag * x)[0].transpose().reshape(-1).tolist(),
        imag=(mag * y)[0].transpose().reshape(-1).tolist(),
        num_frames=mag.shape[2],
    )
    config = knf.StftConfig(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window_type="hann",
        center=True,
        pad_mode="reflect",
        normalized=False,
    )
    istft = knf.IStft(config)
    audio_vocos = istft(stft_result)
    vocos_end_t = dt.datetime.now()

    audio_vocos = np.array(audio_vocos)
    #  audio = audio / 2
    print("vocos max/min", np.max(audio_vocos), np.min(audio_vocos))

    sf.write("vocos.wav", audio_vocos, am.sample_rate, "PCM_16")

    hifigan_start_t = dt.datetime.now()
    audio_hifigan = vocoder(mel)
    hifigan_end_t = dt.datetime.now()
    audio_hifigan = audio_hifigan.squeeze()

    print("hifigan max/min", np.max(audio_hifigan), np.min(audio_hifigan))

    sample_rate = am.sample_rate
    sf.write("hifigan-v2.wav", audio_hifigan, sample_rate, "PCM_16")

    am_t = (mel_end_t - mel_start_t).total_seconds()
    vocos_t = (vocos_end_t - vocos_start_t).total_seconds()
    hifigan_t = (hifigan_end_t - hifigan_start_t).total_seconds()

    mean_audio_duration = (
        (audio_vocos.shape[-1] + audio_hifigan.shape[-1]) / 2 / sample_rate
    )
    rtf_am = am_t / mean_audio_duration

    rtf_vocos = vocos_t * sample_rate / audio_vocos.shape[-1]
    rtf_hifigan = hifigan_t * sample_rate / audio_hifigan.shape[-1]

    print(
        "Audio duration for vocos {:.3f} s".format(audio_vocos.shape[-1] / sample_rate)
    )
    print(
        "Audio duration for hifigan {:.3f} s".format(
            audio_hifigan.shape[-1] / sample_rate
        )
    )
    print("Mean audio duration: {:.3f} s".format(mean_audio_duration))
    print("RTF for acoustic model {:.3f}".format(rtf_am))
    print("RTF for vocos {:.3f}".format(rtf_vocos))
    print("RTF for hifigan {:.3f}".format(rtf_hifigan))


if __name__ == "__main__":
    main()
