#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)


import time
from typing import List

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


def get_token2id():
    ans = dict()
    with open("tokens.txt", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 2:
                token, idx = fields
                ans[token] = int(idx)
            else:
                assert len(fields) == 1, (len(fields), line)
                ans[" "] = int(fields[0])
    return ans


def show(name, sess):
    print(f"----{name} input----")
    for i in sess.get_inputs():
        print(i)

    print(f"----{name} output----")

    for i in sess.get_outputs():
        print(i)

    print()


class OnnxModel:
    def __init__(self, model: str):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts
        self.model = ort.InferenceSession(
            model,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        meta = self.model.get_modelmeta().custom_metadata_map
        print(meta)

        show("model", self.model)

        self.sample_rate = int(meta["sample_rate"])

    def __call__(self, x: List[int], noise_scale=0.667, speed=1.0):
        noise_scale = np.array([noise_scale], dtype=np.float32)
        length_scale = np.array([1 / speed], dtype=np.float32)
        x_length = np.array([len(x)], dtype=np.int64)
        x = np.array(x, dtype=np.int64)[None]

        y = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x,
                self.model.get_inputs()[1].name: x_length,
                self.model.get_inputs()[2].name: noise_scale,
                self.model.get_inputs()[3].name: length_scale,
            },
        )[0]
        return y


def main():
    token2id = get_token2id()
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for s, i in token2id.items():
            f.write(f"{s} {i}\n")

    text = (
        "Today as always, men fall into two groups: slaves and free men."
        + " Whoever does not have two-thirds of his day for himself, "
        + "is a slave, whatever he may be: a statesman, a businessman, "
        + "an official, or a scholar."
    )

    model = OnnxModel(model="model.onnx")
    start = time.time()

    tokens = phonemize_espeak(text, "en-us")
    #  print(text)
    #  print(tokens)
    tokens = sum(tokens, [])  # flatten
    #  print(tokens)

    ids = [token2id.get(t) for t in tokens]
    #  print(ids)
    padded = [0] * (2 * len(ids) + 1)
    padded[1::2] = ids
    #  print(padded)

    waveform = model(padded)

    waveform = waveform[0, 0]
    end = time.time()
    elapsed_seconds = end - start
    audio_duration = len(waveform) / model.sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    sf.write(
        "generated.wav",
        waveform,
        samplerate=model.sample_rate,
        subtype="PCM_16",
    )

    print(" Saved to ./generated.wav")
    print(f" Elapsed seconds: {elapsed_seconds:.3f}")
    print(f" Audio duration in seconds: {audio_duration:.3f}")
    print(f" RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    main()
