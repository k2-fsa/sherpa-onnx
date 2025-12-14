#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import time

import numpy as np
import onnxruntime as ort
import soundfile as sf


def display(sess):
    print("==========Input==========")
    for i in sess.get_inputs():
        print(i)
    print("==========Output==========")
    for i in sess.get_outputs():
        print(i)


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
        logits = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x,
            },
        )[0]
        # [batch_size, T, vocab_size]
        return logits


def load_tokens():
    id2token = dict()
    with open("./tokens.txt", encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) == 1:
                id2token[int(fields[0])] = " "
            else:
                t, idx = fields
                id2token[int(idx)] = t
    return id2token


def load_audio(filename):
    samples, sr = sf.read(filename, always_2d=True, dtype="float32")
    samples = samples[:, 0]  # only use the first channel
    if sr != 16000:
        import librosa

        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
    if len(samples) / 16000 > 40:
        raise ValueError(f"{filename} is too long. Support at most 40 seconds")

    mean = np.mean(samples, axis=0, keepdims=True)
    var = np.var(samples, axis=0, keepdims=True)

    eps = 1e-5
    return (samples - mean) / np.sqrt(var + eps)


def test(filename, wav_file_list, num_iter=1):
    id2token = load_tokens()
    model = OnnxModel(filename)

    for it in range(num_iter):
        for wav in wav_file_list:
            print(f"---test {filename} with {wav}----iter---{it}")
            start = time.time()
            samples = load_audio(wav)

            logits = model(samples[None])
            ids = logits[0].argmax(axis=-1)
            ans = []
            prev = -1
            blank = 0
            for i in ids:
                if i != blank and i != prev:
                    ans.append(i)
                prev = i

            words = [id2token[k] for k in ans]
            end = time.time()
            elapsed_seconds = end - start
            audio_duration = samples.shape[0] / 16000
            real_time_factor = elapsed_seconds / audio_duration

            print("---> text is----", "".join(words))
            print(f"RTF: {real_time_factor}")
            print()


def main():
    wav_file_list = ["./en.wav", "./de.wav", "./es.wav", "./fr.wav"]
    test("./model.onnx", wav_file_list)

    test("./model.int8.onnx", wav_file_list)


if __name__ == "__main__":
    main()
