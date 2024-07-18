#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime
import onnxruntime as ort
import soundfile as sf
import torch


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model.onnx",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--wave",
        type=str,
        required=True,
        help="The input wave to be recognized",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="the language of the input wav file. Supported values: zh, en, ja, ko, yue, auto",
    )

    parser.add_argument(
        "--use-itn",
        type=int,
        default=0,
        help="1 to use inverse text normalization. 0 to not use inverse text normalization",
    )

    return parser.parse_args()


class OnnxModel:
    def __init__(self, filename):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        meta = self.model.get_modelmeta().custom_metadata_map

        self.window_size = int(meta["lfr_window_size"])  # lfr_m
        self.window_shift = int(meta["lfr_window_shift"])  # lfr_n

        lang_zh = int(meta["lang_zh"])
        lang_en = int(meta["lang_en"])
        lang_ja = int(meta["lang_ja"])
        lang_ko = int(meta["lang_ko"])
        lang_auto = int(meta["lang_auto"])

        self.lang_id = {
            "zh": lang_zh,
            "en": lang_en,
            "ja": lang_ja,
            "ko": lang_ko,
            "auto": lang_auto,
        }
        self.with_itn = int(meta["with_itn"])
        self.without_itn = int(meta["without_itn"])

        neg_mean = meta["neg_mean"].split(",")
        neg_mean = list(map(lambda x: float(x), neg_mean))

        inv_stddev = meta["inv_stddev"].split(",")
        inv_stddev = list(map(lambda x: float(x), inv_stddev))

        self.neg_mean = np.array(neg_mean, dtype=np.float32)
        self.inv_stddev = np.array(inv_stddev, dtype=np.float32)

    def __call__(self, x, x_length, language, text_norm):
        logits = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_length.numpy(),
                self.model.get_inputs()[2].name: language.numpy(),
                self.model.get_inputs()[3].name: text_norm.numpy(),
            },
        )[0]

        return torch.from_numpy(logits)


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def load_tokens(filename):
    ans = dict()
    i = 0
    with open(filename, encoding="utf-8") as f:
        for line in f:
            ans[i] = line.strip().split()[0]
            i += 1
    return ans


def compute_feat(
    samples,
    sample_rate,
    neg_mean: np.ndarray,
    inv_stddev: np.ndarray,
    window_size: int = 7,  # lfr_m
    window_shift: int = 6,  # lfr_n
):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    online_fbank.input_finished()

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )
    assert features.data.contiguous is True
    assert features.dtype == np.float32, features.dtype

    T = (features.shape[0] - window_size) // window_shift + 1
    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(T, features.shape[1] * window_size),
        strides=((window_shift * features.shape[1]) * 4, 4),
    )

    features = (features + neg_mean) * inv_stddev

    return features


def main():
    args = get_args()
    print(vars(args))
    samples, sample_rate = load_audio(args.wave)
    if sample_rate != 16000:
        import librosa

        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    model = OnnxModel(filename=args.model)

    features = compute_feat(
        samples=samples,
        sample_rate=sample_rate,
        neg_mean=model.neg_mean,
        inv_stddev=model.inv_stddev,
        window_size=model.window_size,
        window_shift=model.window_shift,
    )

    features = torch.from_numpy(features).unsqueeze(0)
    features_length = torch.tensor([features.size(1)], dtype=torch.int32)

    language = model.lang_id["auto"]
    if args.language in model.lang_id:
        language = model.lang_id[args.language]
    else:
        print(f"Invalid language: '{args.language}'")
        print("Use auto")

    if args.use_itn:
        text_norm = model.with_itn
    else:
        text_norm = model.without_itn

    language = torch.tensor([language], dtype=torch.int32)
    text_norm = torch.tensor([text_norm], dtype=torch.int32)

    logits = model(
        x=features,
        x_length=features_length,
        language=language,
        text_norm=text_norm,
    )

    idx = logits.squeeze(0).argmax(dim=-1)
    # idx is of shape (T,)
    idx = torch.unique_consecutive(idx)

    blank_id = 0
    idx = idx[idx != blank_id].tolist()

    tokens = load_tokens(args.tokens)
    text = "".join([tokens[i] for i in idx])

    text = text.replace("▁", " ")
    print(text)


if __name__ == "__main__":
    main()
