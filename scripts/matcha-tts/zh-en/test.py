#!/usr/bin/env python3

"""
AM

NodeArg(name='x', type='tensor(int64)', shape=['N', 'L'])
NodeArg(name='x_length', type='tensor(int64)', shape=['N'])
NodeArg(name='noise_scale', type='tensor(float)', shape=[1])
NodeArg(name='length_scale', type='tensor(float)', shape=[1])
-----
NodeArg(name='mel', type='tensor(float)', shape=['N', 80, 'L'])

Vocoder

NodeArg(name='mels', type='tensor(float)', shape=['batch_size', 80, 'time'])
-----
NodeArg(name='mag', type='tensor(float)', shape=['batch_size', 'Clipmag_dim_1', 'time'])
NodeArg(name='x', type='tensor(float)', shape=['batch_size', 'Cosx_dim_1', 'time'])
NodeArg(name='y', type='tensor(float)', shape=['batch_size', 'Cosx_dim_1', 'time'])
"""

import argparse

import re

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--am",
        type=str,
        default="./model-steps-3.onnx",
        help="Path to the acoustic model",
    )

    parser.add_argument(
        "--vocoder",
        type=str,
        default="./vocos-16khz-univ.onnx",
        help="Path to the vocoder",
    )
    parser.add_argument(
        "--tokens", type=str, default="./tokens.txt", help="Path to the tokens.txt"
    )

    parser.add_argument(
        "--lexicon", type=str, default="./lexicon.txt", help="Path to the lexicon.txt"
    )

    parser.add_argument(
        "--text",
        type=str,
        #  default="这是一个中英文测试. It can also speak English. 你觉得中英文说的如何呀?",
        default="中英文合成测试. It supports both English 和中文合成",
        help="The text for generation",
    )

    parser.add_argument(
        "--out-wav",
        type=str,
        default="generated.wav",
        help="Path to save the generated wav",
    )
    return parser.parse_args()


def load_tokens(filename: str):
    ans = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 1:
                ans[" "] = int(fields[0])
            else:
                assert len(fields) == 2, (line, fields)
                ans[fields[0]] = int(fields[1])
    return ans


def load_lexicon(filename: str, token2id):
    ans = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            tokens = fields[1:]
            ids = [token2id[t] for t in tokens]
            ans[fields[0]] = ids
    return ans


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
        print(f"vocos {self.model.get_modelmeta().custom_metadata_map}")

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


class OnnxModel:
    def __init__(
        self,
        filename: str,
    ):
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

    def __call__(self, x: np.ndarray):
        assert x.ndim == 2, x.shape
        assert x.shape[0] == 1, x.shape

        x_lengths = np.array([x.shape[1]], dtype=np.int64)

        noise_scale = 1.0
        length_scale = 1.0

        mel = self.model.run(
            [self.model.get_outputs()[0].name],
            {
                self.model.get_inputs()[0].name: x,
                self.model.get_inputs()[1].name: x_lengths,
                self.model.get_inputs()[2].name: np.array(
                    [noise_scale], dtype=np.float32
                ),
                self.model.get_inputs()[3].name: np.array(
                    [length_scale], dtype=np.float32
                ),
            },
        )[0]
        # mel: (batch_size, feat_dim, num_frames)

        return mel


def main():
    args = get_args()
    print(vars(args))
    am = OnnxModel(args.am)
    vocoder = OnnxVocosModel(args.vocoder)

    token2id = load_tokens(args.tokens)
    id2token = {i: t for t, i in token2id.items()}
    lexicon = load_lexicon(args.lexicon, token2id)

    text = args.text

    pattern = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9 ,.!\?]+")

    ids = []
    for match in pattern.finditer(text):
        segment = match.group()
        if segment in token2id:
            print(segment)
            ids.append(token2id[segment])
        elif re.match(r"[\u4e00-\u9fff]+", segment):
            # process chinese
            print(segment)
            for w in segment:
                if w in lexicon:
                    ids += lexicon[w]
                else:
                    print(f"Ignore {w}")
        else:
            print(segment)
            segment = segment.strip()
            tokens_list = phonemize_espeak(segment, "en-us")
            tokens = sum(tokens_list, [])
            for t in tokens:
                ids.append(token2id[t])

    tokens = np.array([ids], dtype=np.int64)
    mel = am(tokens)
    print(tokens)
    print(mel.shape)

    mag, x, y = vocoder(mel)
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

    audio_vocos = np.array(audio_vocos)

    sf.write(args.out_wav, audio_vocos, am.sample_rate, "PCM_16")


if __name__ == "__main__":
    main()
