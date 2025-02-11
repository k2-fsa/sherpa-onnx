#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
female (7)
'af', 'af_bella', 'af_nicole','af_sarah', 'af_sky',
'bf_emma', 'bf_isabella',

male (4)
'am_adam',  'am_michael', 'bm_george', 'bm_lewis'
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

import onnxruntime as ort
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model",
    )

    parser.add_argument(
        "--voices-bin",
        type=str,
        required=True,
        help="Path to the voices.bin",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )
    return parser.parse_args()


def show(filename):
    session_opts = ort.SessionOptions()
    session_opts.log_severity_level = 3
    sess = ort.InferenceSession(filename, session_opts)
    for i in sess.get_inputs():
        print(i)

    print("-----")

    for i in sess.get_outputs():
        print(i)


#  NodeArg(name='tokens', type='tensor(int64)', shape=[1, 'tokens1'])
#  NodeArg(name='style', type='tensor(float)', shape=[1, 256])
#  NodeArg(name='speed', type='tensor(float)', shape=[1])
#  -----
#  NodeArg(name='audio', type='tensor(float)', shape=['audio0'])


def load_tokens(filename: str) -> Dict[str, int]:
    ans = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 2:
                token, idx = fields
                ans[token] = int(idx)
            else:
                assert len(fields) == 1, (len(fields), line)
                ans[" "] = int(fields[0])
    return ans


def load_voices(speaker_names: List[str], dim: List[int], voices_bin: str):
    embedding = (
        np.fromfile(voices_bin, dtype="uint8")
        .view(np.float32)
        .reshape(len(speaker_names), *dim)
    )
    print("embedding.shape", embedding.shape)
    ans = dict()
    for i in range(len(speaker_names)):
        ans[speaker_names[i]] = embedding[i]

    return ans


class OnnxModel:
    def __init__(self, model_filename: str, voices_bin: str, tokens: str):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts
        self.model = ort.InferenceSession(
            model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.token2id = load_tokens(tokens)

        meta = self.model.get_modelmeta().custom_metadata_map
        print(meta)
        dim = list(map(int, meta["style_dim"].split(",")))
        speaker_names = meta["speaker_names"].split(",")

        self.voices = load_voices(
            speaker_names=speaker_names, dim=dim, voices_bin=voices_bin
        )

        self.sample_rate = int(meta["sample_rate"])

        print(list(self.voices.keys()))
        # ['af', 'af_bella', 'af_nicole', 'af_sarah', 'af_sky', 'am_adam',
        # 'am_michael', 'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis']
        # af -> (511, 1, 256)
        self.max_len = self.voices[next(iter(self.voices))].shape[0] - 1

    def __call__(self, text: str, voice):
        tokens = phonemize_espeak(text, "en-us")
        # tokens is List[List[str]]
        # Each sentence is a List[str]
        # len(tokens) == number of sentences

        tokens = sum(tokens, [])  # flatten
        tokens = "".join(tokens)

        tokens = tokens.replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ").replace(
            "kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ"
        )

        tokens = list(tokens)

        token_ids = [self.token2id[i] for i in tokens]
        token_ids = token_ids[: self.max_len]

        style = self.voices[voice][len(token_ids)]

        token_ids = [0, *token_ids, 0]
        token_ids = np.array([token_ids], dtype=np.int64)

        speed = np.array([1.0], dtype=np.float32)

        audio = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: token_ids,
                self.model.get_inputs()[1].name: style,
                self.model.get_inputs()[2].name: speed,
            },
        )[0]
        return audio


def test(model, voice, text) -> np.ndarray:
    pass


def main():
    args = get_args()
    print(vars(args))
    show(args.model)

    #  tokens = phonemize_espeak("how are you doing?", "en-us")
    # [['h', 'ˌ', 'a', 'ʊ', ' ', 'ɑ', 'ː', 'ɹ', ' ', 'j', 'u', 'ː', ' ', 'd', 'ˈ', 'u', 'ː', 'ɪ', 'ŋ', '?']]
    m = OnnxModel(
        model_filename=args.model, voices_bin=args.voices_bin, tokens=args.tokens
    )

    text = (
        "Today as always, men fall into two groups: slaves and free men."
        + " Whoever does not have two-thirds of his day for himself, "
        + "is a slave, whatever he may be: a statesman, a businessman, "
        + "an official, or a scholar."
    )

    for i, voice in enumerate(m.voices.keys(), 1):
        print(f"Testing {i}/{len(m.voices)} - {voice}/{args.model}")

        start = time.time()
        audio = m(text, voice=voice)
        end = time.time()

        elapsed_seconds = end - start
        audio_duration = len(audio) / m.sample_rate
        real_time_factor = elapsed_seconds / audio_duration

        filename = f"{Path(args.model).stem}-{voice}.wav"
        sf.write(
            filename,
            audio,
            samplerate=m.sample_rate,
            subtype="PCM_16",
        )
        print(f" Saved to {filename}")
        print(f" Elapsed seconds: {elapsed_seconds:.3f}")
        print(f" Audio duration in seconds: {audio_duration:.3f}")
        print(
            f" RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
        )


if __name__ == "__main__":
    main()
