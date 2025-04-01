#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

# Please run this file on your rk3588 board

try:
    from rknnlite.api import RKNNLite
except:
    print("Please run this file on your board (linux + aarch64 + npu)")
    print("You need to install rknn_toolkit_lite2")
    print(
        " from https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages"
    )
    print(
        "https://github.com/airockchip/rknn-toolkit2/blob/v2.1.0/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.1.0-cp310-cp310-linux_aarch64.whl"
    )
    print("is known to work")
    raise

import soundfile as sf
import time
from typing import Tuple
import numpy as np
from pathlib import Path


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel

    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def init_model(filename, target_platform="rk3588"):
    if not Path(filename).is_file():
        exit(f"{filename} does not exist")

    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(path=filename)
    if ret != 0:
        exit(f"Load model {filename} failed!")

    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        exit(f"Failed to init rknn runtime for {filename}")
    return rknn_lite


class RKNNModel:
    def __init__(self, model: str, target_platform="rk3588"):
        self.model = init_model(model)

    def release(self):
        self.model.release()

    def __call__(self, x: np.ndarray, h: np.ndarray, c: np.ndarray):
        """
        Args:
          x: (1, 512), np.float32
          h: (2, 1, 64), np.float32
          c: (2, 1, 64), np.float32
        Returns:
          prob:
          next_h:
          next_c
        """
        out, next_h, next_c = self.model.inference(inputs=[x, h, c])
        return out.item(), next_h, next_c


def main():
    model = RKNNModel(model="./m.rknn")
    for i in range(1):
        test(model)


def test(model):
    print("started")
    start = time.time()
    samples, sample_rate = load_audio("./lei-jun-test.wav")
    assert sample_rate == 16000, sample_rate

    window_size = 512

    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)

    threshold = 0.5
    num_windows = samples.shape[0] // window_size
    out = []
    for i in range(num_windows):
        print(i, num_windows)
        this_samples = samples[i * window_size : (i + 1) * window_size]
        # print("at input", k, np.sum(this_samples), np.mean(this_samples))
        # print("h", np.sum(h), np.mean(h))
        # print("c", np.sum(c), np.mean(c))
        prob, h, c = model(this_samples[None], h, c)
        # print("at output", k, prob, np.sum(h), np.mean(h), np.sum(c), np.mean(c))
        out.append(prob > threshold)

    min_speech_duration = 0.25 * sample_rate / window_size
    min_silence_duration = 0.25 * sample_rate / window_size

    result = []
    last = -1
    for k, f in enumerate(out):
        if f >= threshold:
            if last == -1:
                last = k
        elif last != -1:
            if k - last > min_speech_duration:
                result.append((last, k))
            last = -1

    if last != -1 and k - last > min_speech_duration:
        result.append((last, k))

    if not result:
        print("Empty for ./lei-jun-test.wav")
        return

    print(result)

    final = [result[0]]
    for r in result[1:]:
        f = final[-1]
        if r[0] - f[1] < min_silence_duration:
            final[-1] = (f[0], r[1])
        else:
            final.append(r)

    for f in final:
        start = f[0] * window_size / sample_rate
        end = f[1] * window_size / sample_rate
        print("{:.3f} -- {:.3f}".format(start, end))


if __name__ == "__main__":
    main()
