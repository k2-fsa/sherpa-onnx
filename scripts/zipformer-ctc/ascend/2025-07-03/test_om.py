#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from ais_bench.infer.interface import InferSession

from onnx_test import BCHAR_TO_BYTE, compute_feat, load_audio, load_tokens


class OmModel:
    def __init__(self):
        self.model = InferSession(device_id=0, model_path="./model.om", debug=False)

        self.max_len = self.model.get_inputs()[0].shape[1]
        print("---model---")
        for i in self.model.get_inputs():
            print(i.name, i.datatype, i.shape)

        print("-----")

        for i in self.model.get_outputs():
            print(i.name, i.datatype, i.shape)

    def __call__(self, x):
        """
        Args:
          x: (N, T, C)
        Returns:
          log_probs: (N, T, vocab_size)
        """
        return self.model.infer([x], mode="static", custom_sizes=10000000)[0]


def main():
    samples, sample_rate = load_audio("./test_wavs/0.wav")
    model = OmModel()

    features = compute_feat(
        samples=samples, sample_rate=sample_rate, max_len=model.max_len
    )
    print("features.shape", features.shape)

    log_probs = model(x=features[None])
    print("log_probs.shape", log_probs.shape, type(log_probs))

    idx = log_probs[0].argmax(axis=-1)
    print("idx", idx)
    print(len(idx))
    prev = -1
    ids = []
    for i in idx:
        if i != prev:
            ids.append(i)
        prev = i
    ids = [i for i in ids if i != 0]
    print(ids)

    tokens = load_tokens("./tokens.txt")
    text = "".join([tokens[i] for i in ids])

    s = b""
    for t in text:
        if t == "▁":
            continue
        elif t in BCHAR_TO_BYTE:
            s += bytes([BCHAR_TO_BYTE[t]])
        else:
            print("skip OOV", t)

    print(s.decode())


if __name__ == "__main__":
    main()
