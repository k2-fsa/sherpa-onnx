#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)


import re
import time
from typing import Dict, List

import jieba
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from misaki import zh

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )


name = "bm_fable"


def show(filename):
    session_opts = ort.SessionOptions()
    session_opts.log_severity_level = 3
    sess = ort.InferenceSession(filename, session_opts)
    for i in sess.get_inputs():
        print(i)

    print("-----")

    for i in sess.get_outputs():
        print(i)


"""
NodeArg(name='tokens', type='tensor(int64)', shape=[1, 'sequence_length'])
NodeArg(name='style', type='tensor(float)', shape=[1, 256])
NodeArg(name='speed', type='tensor(float)', shape=[1])
-----
NodeArg(name='audio', type='tensor(float)', shape=['audio_length'])
"""


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


def load_lexicon(filename: str) -> Dict[str, List[str]]:
    ans = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            w, tokens = line.strip().split(" ", maxsplit=1)
            ans[w] = "".join(tokens.split())
    return ans


class OnnxModel:
    def __init__(self, model_filename: str, tokens: str, lexicon: str):
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
        self.word2tokens = load_lexicon(lexicon)
        self.voices = torch.load("./af_bella.pt", weights_only=True).numpy()
        self.voices = torch.load(f"./{name}.pt", weights_only=True).numpy()
        # self.voices: (510, 1, 256)
        print(self.voices.shape)

        self.sample_rate = 24000

        self.max_len = self.voices.shape[0]

    def __call__(self, text: str):
        punctuations = ';:,.!?-…()"“”'
        text = text.lower()
        g2p = zh.ZHG2P()

        tokens = ""

        for t in re.findall("[\u4E00-\u9FFF]+|[\u0000-\u007f]+", text):
            if ord(t[0]) < 0x7F:
                for w in t.split():
                    while w:
                        if w[0] in punctuations:
                            tokens += w[0] + " "
                            w = w[1:]
                            continue

                        if w[-1] in punctuations:
                            if w[:-1] in self.word2tokens:
                                tokens += self.word2tokens[w[:-1]]
                                tokens += w[-1]
                        else:
                            if w in self.word2tokens:
                                tokens += self.word2tokens[w]
                            else:
                                print(f"Use espeak-ng for word {w}")
                                tokens += "".join(phonemize_espeak(w, "en-us")[0])

                        tokens += " "
                        break
            else:
                # Chinese
                for w in jieba.cut(t):
                    if w in self.word2tokens:
                        tokens += self.word2tokens[w]
                    else:
                        for i in w:
                            if i in self.word2tokens:
                                tokens += self.word2tokens[i]
                            else:
                                print(f"skip {i}")

        token_ids = [self.token2id[i] for i in tokens]
        token_ids = token_ids[: self.max_len]

        style = self.voices[len(token_ids)]

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


def main():
    m = OnnxModel(
        model_filename="./kokoro.onnx",
        tokens="./tokens.txt",
        lexicon="./lexicon.txt",
    )
    text = "来听一听, 这个是什么口音? How are you doing? Are you ok? Thank you! 你觉得中英文说得如何呢?"

    text = text.lower()

    start = time.time()
    audio = m(text)
    end = time.time()

    elapsed_seconds = end - start
    audio_duration = len(audio) / m.sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    filename = f"kokoro_v1.0_{name}_zh_en.wav"
    sf.write(
        filename,
        audio,
        samplerate=m.sample_rate,
        subtype="PCM_16",
    )
    print(f" Saved to {filename}")
    print(f" Elapsed seconds: {elapsed_seconds:.3f}")
    print(f" Audio duration in seconds: {audio_duration:.3f}")
    print(f" RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    main()
