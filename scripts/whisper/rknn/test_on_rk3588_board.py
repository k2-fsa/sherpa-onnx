#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
usage:

./test_on_rk3588_board.py  --encoder ./base-encoder.rknn --decoder ./base-decoder.rknn --tokens ./base-tokens.txt --wav ./en-16k.wav

./test_on_rk3588_board.py  --encoder ./base.en-encoder.rknn --decoder ./base.en-decoder.rknn --tokens ./base.en-tokens.txt --wav ./en-16k.wav
"""

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

import argparse
import base64
import time
from pathlib import Path
from typing import List, Tuple

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to the encoder",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Path to the decoder",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to the tokens",
    )

    parser.add_argument(
        "--wav",
        type=str,
        required=True,
        help="Path to the test wav",
    )

    return parser.parse_args()


def causal_mask_1d(n: int, L: int):
    """
    Returns a 1-D int mask of shape (L,) with:
      0 -> allowed
      1 -> masked (will be converted to -inf later)
    """
    mask = np.ones((L,), dtype=np.int32)
    if n > 0:
        mask[:n] = 0
    return mask


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel

    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_features(samples: np.ndarray, dim: int = 80) -> np.ndarray:
    """
    Returns:
      Return a 1-D float32 tensor of shape (1, 80, 3000) containing the features.
    """
    features = []
    opts = knf.WhisperFeatureOptions()
    opts.dim = dim
    online_whisper_fbank = knf.OnlineWhisperFbank(opts)
    online_whisper_fbank.accept_waveform(16000, samples)
    online_whisper_fbank.input_finished()
    for i in range(online_whisper_fbank.num_frames_ready):
        f = online_whisper_fbank.get_frame(i)
        f = torch.from_numpy(f)
        features.append(f)

    features = torch.stack(features)

    log_spec = torch.clamp(features, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0
    # mel (T, 80)

    # We pad 1500 frames at the end so that it is able to detect eot
    # You can use another value instead of 1500.
    mel = torch.nn.functional.pad(mel, (0, 0, 0, 1500), "constant", 0)
    # Note that if it throws for a multilingual model,
    # please use a larger value, say 300

    target = 3000
    if mel.shape[0] > target:
        # -50 so that there are some zero tail paddings.
        mel = mel[: target - 50]
        mel = torch.nn.functional.pad(mel, (0, 0, 0, 50), "constant", 0)
    elif mel.shape[0] < target:
        mel = torch.nn.functional.pad(
            mel, (0, 0, 0, target - mel.shape[0]), "constant", 0
        )

    mel = mel.t().unsqueeze(0)

    return mel


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


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
    def __init__(
        self,
        encoder: str,
        decoder: str,
        sot_sequence: List[int],
        eot: int,
        n_text_layer: int,
        n_text_ctx: int,
        n_text_state: int,
        target_platform="rk3588",
    ):
        self.sot_sequence = sot_sequence
        self.eot = eot
        self.n_text_layer = n_text_layer
        self.n_text_ctx = n_text_ctx
        self.n_text_state = n_text_state

        print("sot_sequence", self.sot_sequence)
        print("eot", self.eot)

        self.encoder = init_model(encoder)
        self.decoder = init_model(decoder)

    def release(self):
        self.encoder.release()
        self.decoder.release()

    def run_encoder(self, x: np.ndarray):
        """
        Args:
          x: (1, 80, 3000), np.float32
        Returns:
          cross_kv:
           - (k, v) for layer 0
           - (k, v) for layer 1
           - (k, v) for layer 2
           - (k, v) for layer 3
        """
        out = self.encoder.inference(inputs=[x.numpy()])
        print("after running encoder", len(out))
        return out

    def get_self_cache(self) -> List[np.ndarray]:
        self_cache = []
        batch_size = 1
        for i in range(self.n_text_layer):
            k = np.zeros(
                (batch_size, self.n_text_ctx, self.n_text_state), dtype=np.float32
            )
            v = np.zeros(
                (batch_size, self.n_text_ctx, self.n_text_state), dtype=np.float32
            )
            self_cache.extend([k, v])
        return self_cache

    def run_decoder(self, tokens: np.ndarray, self_kv, cross_kv, offset, mask):
        """
        Args:
          tokens: (1, 1), np.int32
          offset: (1,), np.int32
          mask: (model.n_text_ctx,), np.int32
        Returns:
          logit: (1, 1, vocab_size)
          this_self_kv
        """
        return self.decoder.inference(
            inputs=[tokens] + self_kv + cross_kv + [offset, mask]
        )


def main():
    args = get_args()
    print(vars(args))

    id2token = load_tokens(args.tokens)

    if ".en" in args.encoder:
        print("here", args.encoder)
        sot_sequence = [50257, 50362]
        eot = 50256
    else:
        print("not here", args.encoder)
        sot_sequence = [50258, 50259, 50359, 50363]
        eot = 50257

    if "tiny" in args.encoder:
        n_text_layer = 4
        n_text_ctx = 448
        n_text_state = 384
    elif "base" in args.encoder:
        n_text_layer = 6
        n_text_ctx = 448
        n_text_state = 512
    elif "small" in args.encoder:
        n_text_layer = 12
        n_text_ctx = 448
        n_text_state = 768
    elif "medium" in args.encoder:
        n_text_layer = 24
        n_text_ctx = 448
        n_text_state = 1024
    else:
        assert False, f"Unsupported encoder {args.encoder}"

    model = RKNNModel(
        encoder=args.encoder,
        decoder=args.decoder,
        sot_sequence=sot_sequence,
        eot=eot,
        n_text_layer=n_text_layer,
        n_text_ctx=n_text_ctx,
        n_text_state=n_text_state,
    )

    for i in range(1):
        test(model, id2token)


def test(model, id2token):

    start = time.time()
    samples, sample_rate = load_audio("./en-16k.wav")
    assert sample_rate == 16000, sample_rate

    features = compute_features(samples)
    print(features.shape)
    cross_kv = model.run_encoder(features)

    self_kv = model.get_self_cache()

    offset = np.array([0], dtype=np.int32)
    for t in model.sot_sequence:
        token = np.array([[t]], dtype=np.int32)  # sot
        mask = causal_mask_1d(offset.item(), model.n_text_ctx)
        print(t, model.sot_sequence, token, mask.shape, len(cross_kv), len(self_kv))

        out = model.run_decoder(
            tokens=token, self_kv=self_kv, cross_kv=cross_kv, offset=offset, mask=mask
        )

        for i in range(1, len(out)):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]

        offset += 1

    idx = out[0][0, 0].argmax()

    eot = model.eot

    ans = []

    while idx != eot and offset.item() < 100:
        ans.append(idx)
        token = np.array([[idx]], dtype=np.int32)

        mask = causal_mask_1d(offset.item(), model.n_text_ctx)

        out = model.run_decoder(
            tokens=token, self_kv=self_kv, cross_kv=cross_kv, offset=offset, mask=mask
        )

        for i in range(1, len(out)):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]

        offset += 1
        idx = out[0][0, 0].argmax()

    print(ans)

    s = b""
    for i in ans:
        if i in id2token:
            s += base64.b64decode(id2token[i])

    print(s.decode().strip())


if __name__ == "__main__":
    main()
