#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import kaldi_native_fbank as knf
import librosa
import numpy as np
from ais_bench.infer.interface import InferSession


def compute_feat(filename):
    sample_rate = 16000
    samples, _ = librosa.load(filename, sr=sample_rate)
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
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
    print("features sum", features.sum(), features.shape)

    window_size = 7  # lfr_m
    window_shift = 6  # lfr_n

    T = (features.shape[0] - window_size) // window_shift + 1
    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(T, features.shape[1] * window_size),
        strides=((window_shift * features.shape[1]) * 4, 4),
    )
    return np.copy(features)


def load_tokens():
    ans = dict()
    i = 0
    with open("tokens.txt", encoding="utf-8") as f:
        for line in f:
            ans[i] = line.strip().split()[0]
            i += 1
    return ans


class OmModel:
    def __init__(self):
        print("init encoder")
        self.encoder = InferSession(device_id=0, model_path="./encoder.om", debug=False)
        self.decoder = InferSession(device_id=1, model_path="./decoder.om", debug=False)
        self.predictor = InferSession(
            device_id=0, model_path="./predictor.om", debug=False
        )

        print("---encoder---")
        for i in self.encoder.get_inputs():
            print(i.name, i.datatype, i.shape)

        print("-----")

        for i in self.encoder.get_outputs():
            print(i.name, i.datatype, i.shape)

        print("---decoder---")
        for i in self.decoder.get_inputs():
            print(i.name, i.datatype, i.shape)

        print("-----")

        for i in self.decoder.get_outputs():
            print(i.name, i.datatype, i.shape)

        print("---predictor---")
        for i in self.predictor.get_inputs():
            print(i.name, i.datatype, i.shape)

        print("-----")

        for i in self.predictor.get_outputs():
            print(i.name, i.datatype, i.shape)

    def run_encoder(self, features):
        encoder_out = self.encoder.infer([features], mode="dymshape")[0]
        return encoder_out

    def run_predictor(self, encoder_out):
        alphas = self.predictor.infer([encoder_out], mode="dymshape")[0]
        return alphas

    def run_decoder(self, encoder_out, acoustic_embedding):
        decoder_out = self.decoder.infer(
            [encoder_out, acoustic_embedding], mode="dymshape"
        )[0]
        return decoder_out


def get_acoustic_embedding(alpha: np.array, hidden: np.array):
    """
    Args:
      alpha: (T,)
      hidden: (T, C)
    Returns:
      acoustic_embeds: (num_tokens, C)
    """
    alpha = alpha.tolist()
    acc = 0

    embeddings = []
    cur_embedding = np.zeros((hidden.shape[1],), dtype=np.float32)

    for i, w in enumerate(alpha):
        acc += w
        if acc >= 1:
            overflow = acc - 1
            remain = w - overflow
            cur_embedding += remain * hidden[i]
            embeddings.append(cur_embedding)

            cur_embedding = overflow * hidden[i]
            acc = overflow
        else:
            cur_embedding += w * hidden[i]

    if len(embeddings) == 0:
        raise ValueError("No speech in the audio file")

    embeddings = np.array(embeddings)
    return embeddings


def main():
    features = compute_feat("./test_wavs/1.wav")
    print("here", features.shape, features.shape[0] > 83)

    print("features.shape", features.shape)

    print("sum", features.sum(), features.mean())

    model = OmModel()

    encoder_out = model.run_encoder(features[None])
    print("encoder_out.shape", encoder_out.shape)
    print("encoder_out.sum", encoder_out.sum(), encoder_out.mean())

    alpha = model.run_predictor(encoder_out)
    print("alpha.shape", alpha.shape)
    print("alpha.sum()", alpha.sum(), alpha.mean())

    acoustic_embedding = get_acoustic_embedding(alpha[0], encoder_out[0])
    print("acoustic_embedding.shape", acoustic_embedding.shape)
    num_tokens = acoustic_embedding.shape[0]
    print("num_tokens", num_tokens)

    print("acoustic_embedding.sum", acoustic_embedding.sum(), acoustic_embedding.mean())

    decoder_out = model.run_decoder(encoder_out, acoustic_embedding[None])
    print("decoder_out", decoder_out.shape)
    print("decoder_out.sum", decoder_out.sum(), decoder_out.mean())
    yseq = decoder_out[0, :num_tokens].argmax(axis=-1).tolist()
    print(yseq, "-->", len(yseq))

    tokens = load_tokens()
    words = [tokens[i] for i in yseq if i not in (1, 2)]
    print(words)
    text = "".join(words)
    print(text)


if __name__ == "__main__":
    main()
