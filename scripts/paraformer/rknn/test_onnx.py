#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import kaldi_native_fbank as knf
import onnxruntime as ort
import librosa
import torch
import numpy as np


class SinusoidalPositionEncoder(torch.nn.Module):
    def encode(
        self,
        positions: torch.Tensor = None,
        depth: int = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
          positions: (batch_size, )
        """
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(
            torch.tensor([10000], dtype=dtype, device=device)
        ) / (depth / 2 - 1)
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype)
            * (-log_timescale_increment)
        )
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1]
        )
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, batch_size, timesteps, input_dim):
        positions = torch.arange(1, timesteps + 1)[None, :]
        position_encoding = self.encode(positions, input_dim, torch.float32)

        return position_encoding


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


class OnnxModel:
    def __init__(self):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        print("init encoder")
        self.encoder = ort.InferenceSession(
            "./encoder-5-seconds.onnx",
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        print("init decoder")
        self.decoder = ort.InferenceSession(
            "./decoder-5-seconds.onnx",
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        print("init predictor")
        self.predictor = ort.InferenceSession(
            "./predictor-5-seconds.onnx",
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        print("---encoder---")
        for i in self.encoder.get_inputs():
            print(i)

        print("-----")

        for i in self.encoder.get_outputs():
            print(i)

        print("---decoder---")
        for i in self.decoder.get_inputs():
            print(i)

        print("-----")

        for i in self.decoder.get_outputs():
            print(i)

        print("---predictor---")
        for i in self.predictor.get_inputs():
            print(i)

        print("-----")

        for i in self.predictor.get_outputs():
            print(i)

    #  def run_encoder(self, features, pos_emb):
    def run_encoder(self, features):
        (encoder_out,) = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
            ],
            {
                self.encoder.get_inputs()[0].name: features,
                #  self.encoder.get_inputs()[1].name: pos_emb,
            },
        )
        return encoder_out

    def run_predictor(self, encoder_out):
        (alphas,) = self.predictor.run(
            [
                self.predictor.get_outputs()[0].name,
            ],
            {
                self.predictor.get_inputs()[0].name: encoder_out,
            },
        )
        return alphas

    #  def run_decoder(self, encoder_out, acoustic_embedding, mask):
    def run_decoder(self, encoder_out, acoustic_embedding, mask):
        print(
            self.decoder.get_outputs()[0].name,
            self.decoder.get_inputs()[0].name,
            self.decoder.get_inputs()[1].name,
        )
        (decoder_out,) = self.decoder.run(
            [
                self.decoder.get_outputs()[0].name,
            ],
            {
                self.decoder.get_inputs()[0].name: encoder_out,
                self.decoder.get_inputs()[1].name: acoustic_embedding,
                self.decoder.get_inputs()[2].name: mask,
            },
        )
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
    num_tokens = 0

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
    features = compute_feat("./1.wav")
    print("here", features.shape, features.shape[0] > 83)
    if features.shape[0] >= 83:
        features = features[:83]
    else:
        padding = features[-(83 - features.shape[0]) :]
        print("padding", features.shape, padding.shape)
        features = np.concatenate([features, padding])

    pos_emb = (
        SinusoidalPositionEncoder()(1, features.shape[0], features.shape[1])
        .squeeze(0)
        .numpy()
    )

    print("features.shape", features.shape, pos_emb.shape)

    print("sum", features.sum(), features.mean(), pos_emb.sum(), pos_emb.mean())

    model = OnnxModel()

    #  encoder_out = model.run_encoder(features[None], pos_emb[None])
    encoder_out = model.run_encoder(features[None])
    print("encoder_out.shape", encoder_out.shape)
    print("encoder_out.sum", encoder_out.sum(), encoder_out.mean())

    alpha = model.run_predictor(encoder_out)
    print("alpha.shape", alpha.shape)
    print("alpha.sum()", alpha.sum(), alpha.mean())

    acoustic_embedding = get_acoustic_embedding(alpha[0], encoder_out[0])
    print("acoustic_embedding.shape", acoustic_embedding.shape)
    num_tokens = acoustic_embedding.shape[0]

    padding = np.zeros((83 - acoustic_embedding.shape[0], 512), dtype=np.float32)
    print("padding.shape", padding.shape, acoustic_embedding.shape)

    acoustic_embedding = np.concatenate([acoustic_embedding, padding], axis=0)
    print("acoustic_embedding.shape", acoustic_embedding.shape)
    print("acoustic_embedding.sum", acoustic_embedding.sum(), acoustic_embedding.mean())

    mask = np.zeros((83,), dtype=np.float32)
    mask[:num_tokens] = 1
    print(mask)

    decoder_out = model.run_decoder(encoder_out, acoustic_embedding[None], mask)
    #  decoder_out = model.run_decoder(encoder_out, acoustic_embedding[None])
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
