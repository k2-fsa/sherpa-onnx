#!/usr/bin/env python3

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

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

        for i in self.model.get_inputs():
            print(i)

        print("-----")

        for i in self.model.get_outputs():
            print(i)

    def __call__(self, x: torch.tensor):
        assert x.ndim == 3, x.shape
        assert x.shape[0] == 1, x.shape

        mag, x, y = self.model.run(
            [
                self.model.get_outputs()[0].name,
                self.model.get_outputs()[1].name,
                self.model.get_outputs()[2].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
            },
        )
        # audio: (batch_size, num_samples)

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

        for i in self.model.get_inputs():
            print(i)

        print("-----")

        for i in self.model.get_outputs():
            print(i)

    def __call__(self, x: torch.tensor):
        assert x.ndim == 3, x.shape
        assert x.shape[0] == 1, x.shape

        audio = self.model.run(
            [self.model.get_outputs()[0].name],
            {
                self.model.get_inputs()[0].name: x.numpy(),
            },
        )[0]
        # audio: (batch_size, num_samples)

        return torch.from_numpy(audio)


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
        session_opts.intra_op_num_threads = 2

        self.session_opts = session_opts
        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        print(f"{self.model.get_modelmeta().custom_metadata_map}")
        metadata = self.model.get_modelmeta().custom_metadata_map
        self.sample_rate = int(metadata["sample_rate"])

        for i in self.model.get_inputs():
            print(i)

        print("-----")

        for i in self.model.get_outputs():
            print(i)

    def __call__(self, x: torch.tensor):
        assert x.ndim == 2, x.shape
        assert x.shape[0] == 1, x.shape

        x_lengths = torch.tensor([x.shape[1]], dtype=torch.int64)
        print("x_lengths", x_lengths)
        print("x", x.shape)

        noise_scale = torch.tensor([1.0], dtype=torch.float32)
        length_scale = torch.tensor([1.0], dtype=torch.float32)

        mel = self.model.run(
            [self.model.get_outputs()[0].name],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_lengths.numpy(),
                self.model.get_inputs()[2].name: noise_scale.numpy(),
                self.model.get_inputs()[3].name: length_scale.numpy(),
            },
        )[0]
        # mel: (batch_size, feat_dim, num_frames)

        return torch.from_numpy(mel)


def main():
    am = OnnxModel(
        filename="./matcha-icefall-en_US-ljspeech/model-steps-3.onnx",
        tokens="./matcha-icefall-en_US-ljspeech/tokens.txt",
    )
    vocoder = OnnxHifiGANModel("./hifigan_v2.onnx")
    vocos = OnnxVocosModel("./mel_spec_22khz_univ.onnx")

    text = "how are you doing"
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

    print(tokens)
    print(token_ids)

    token_ids2 = [am.token2id["_"]] * (len(token_ids) * 2 + 1)
    token_ids2[1::2] = token_ids
    token_ids = token_ids2
    x = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0)
    mel = am(x)
    print("mel", mel.shape)
    # mel:(1, 80, 78)

    mag, x, y = vocos(mel)
    print(mag.shape)
    print(x.shape)
    print(y.shape)
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
    audio = istft(stft_result)
    audio = np.array(audio)
    #  audio = audio / 2
    print(np.max(audio), np.min(audio))
    sf.write("vocos.wav", np.array(audio, dtype=np.float32), am.sample_rate, "PCM_16")

    audio = vocoder(mel).squeeze().numpy()
    print("audio", audio.shape)
    print(np.max(audio), np.min(audio))

    sample_rate = am.sample_rate
    sf.write("hifigan-v2.wav", audio, sample_rate, "PCM_16")


if __name__ == "__main__":
    main()
