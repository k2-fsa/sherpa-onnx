#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import time
from pathlib import Path
from typing import List

import kaldi_native_fbank as knf
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder", type=str, required=True, help="Path to encoder.onnx"
    )
    parser.add_argument(
        "--decoder", type=str, required=True, help="Path to decoder.onnx"
    )

    parser.add_argument("--tokens", type=str, required=True, help="Path to tokens.txt")

    parser.add_argument(
        "--source-lang",
        type=str,
        help="Language of the input wav. Valid values are: en, de, es, fr",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        help="Language of the recognition result. Valid values are: en, de, es, fr",
    )
    parser.add_argument(
        "--use-pnc",
        type=int,
        default=1,
        help="1 to enable cases and punctuations. 0 to disable that",
    )

    parser.add_argument("--wav", type=str, required=True, help="Path to test.wav")

    return parser.parse_args()


def display(sess, model):
    print(f"=========={model} Input==========")
    for i in sess.get_inputs():
        print(i)
    print(f"=========={model }Output==========")
    for i in sess.get_outputs():
        print(i)


class OnnxModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
    ):
        self.init_encoder(encoder)
        display(self.encoder, "encoder")

        self.init_decoder(decoder)
        display(self.decoder, "decoder")

    def init_encoder(self, encoder):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.normalize_type = meta["normalize_type"]
        print(meta)

    def init_decoder(self, decoder):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    def run_encoder(self, x: np.ndarray, x_lens: np.ndarray):
        """
        Args:
          x: (N, T, C), np.float
          x_lens: (N,), np.int64
        Returns:
          enc_states: (N, T, C)
          enc_lens: (N,), np.int64
          enc_masks: (N, T), np.bool
        """
        enc_states, enc_lens, enc_masks = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
                self.encoder.get_outputs()[2].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x,
                self.encoder.get_inputs()[1].name: x_lens,
            },
        )
        return enc_states, enc_lens, enc_masks

    def run_decoder(
        self,
        decoder_input_ids: np.ndarray,
        decoder_mems_list: List[np.ndarray],
        enc_states: np.ndarray,
        enc_mask: np.ndarray,
    ):
        """
        Args:
          decoder_input_ids: (N, num_tokens), int32
          decoder_mems_list: a list of tensors, each of which is (N, num_tokens, C)
          enc_states: (N, T, C), float
          enc_mask: (N, T), bool
        Returns:
          logits: (1, 1, vocab_size), float
          new_decoder_mems_list:
        """
        (logits, *new_decoder_mems_list) = self.decoder.run(
            [
                self.decoder.get_outputs()[0].name,
                self.decoder.get_outputs()[1].name,
                self.decoder.get_outputs()[2].name,
                self.decoder.get_outputs()[3].name,
                self.decoder.get_outputs()[4].name,
                self.decoder.get_outputs()[5].name,
                self.decoder.get_outputs()[6].name,
            ],
            {
                self.decoder.get_inputs()[0].name: decoder_input_ids,
                self.decoder.get_inputs()[1].name: decoder_mems_list[0],
                self.decoder.get_inputs()[2].name: decoder_mems_list[1],
                self.decoder.get_inputs()[3].name: decoder_mems_list[2],
                self.decoder.get_inputs()[4].name: decoder_mems_list[3],
                self.decoder.get_inputs()[5].name: decoder_mems_list[4],
                self.decoder.get_inputs()[6].name: decoder_mems_list[5],
                self.decoder.get_inputs()[7].name: enc_states,
                self.decoder.get_inputs()[8].name: enc_mask,
            },
        )
        return logits, new_decoder_mems_list


def create_fbank():
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.remove_dc_offset = False
    opts.frame_opts.window_type = "hann"

    opts.mel_opts.low_freq = 0
    opts.mel_opts.num_bins = 128

    opts.mel_opts.is_librosa = True

    fbank = knf.OnlineFbank(opts)
    return fbank


def compute_features(audio, fbank):
    assert len(audio.shape) == 1, audio.shape
    fbank.accept_waveform(16000, audio)
    ans = []
    processed = 0
    while processed < fbank.num_frames_ready:
        ans.append(np.array(fbank.get_frame(processed)))
        processed += 1
    ans = np.stack(ans)
    return ans


def main():
    args = get_args()
    assert Path(args.encoder).is_file(), args.encoder
    assert Path(args.decoder).is_file(), args.decoder
    assert Path(args.tokens).is_file(), args.tokens
    assert Path(args.wav).is_file(), args.wav

    print(vars(args))

    id2token = dict()
    token2id = dict()
    with open(args.tokens, encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                t, idx = fields[0], int(fields[1])
                if line[0] == " ":
                    t = " " + t
            else:
                t = " "
                idx = int(fields[0])

            id2token[idx] = t
            token2id[t] = idx

    model = OnnxModel(args.encoder, args.decoder)

    fbank = create_fbank()

    start = time.time()
    audio, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel
    if sample_rate != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )
        sample_rate = 16000

    features = compute_features(audio, fbank)
    if model.normalize_type != "":
        assert model.normalize_type == "per_feature", model.normalize_type
        mean = features.mean(axis=0, keepdims=True)
        stddev = features.std(axis=0, keepdims=True) + 1e-5
        features = (features - mean) / stddev

    features = np.expand_dims(features, axis=0)
    # features.shape: (1, 291, 128)

    features_len = np.array([features.shape[1]], dtype=np.int64)

    enc_states, _, enc_masks = model.run_encoder(features, features_len)

    decoder_input_ids = []
    decoder_input_ids.append(token2id["<|startofcontext|>"])
    decoder_input_ids.append(token2id["<|startoftranscript|>"])
    decoder_input_ids.append(token2id["<|emo:undefined|>"])
    if args.source_lang in ("en", "es", "de", "fr"):
        decoder_input_ids.append(token2id[f"<|{args.source_lang}|>"])
    else:
        decoder_input_ids.append(token2id[f"<|en|>"])

    if args.target_lang in ("en", "es", "de", "fr"):
        decoder_input_ids.append(token2id[f"<|{args.target_lang}|>"])
    else:
        decoder_input_ids.append(token2id[f"<|en|>"])

    if args.use_pnc:
        decoder_input_ids.append(token2id[f"<|pnc|>"])
    else:
        decoder_input_ids.append(token2id[f"<|nopnc|>"])

    decoder_input_ids.append(token2id[f"<|noitn|>"])
    decoder_input_ids.append(token2id["<|notimestamp|>"])
    decoder_input_ids.append(token2id["<|nodiarize|>"])

    decoder_mems_list = [np.zeros((1, 0, 1024), dtype=np.float32) for _ in range(6)]

    for pos, decoder_input_id in enumerate(decoder_input_ids):
        logits, decoder_mems_list = model.run_decoder(
            np.array([[decoder_input_id, pos]], dtype=np.int32),
            decoder_mems_list,
            enc_states,
            enc_masks,
        )
    tokens = [logits.argmax()]
    print("decoder_input_ids", decoder_input_ids)
    eos = token2id["<|endoftext|>"]

    for i in range(1, 200):
        decoder_input_ids = [tokens[-1], i]
        logits, decoder_mems_list = model.run_decoder(
            np.array([decoder_input_ids], dtype=np.int32),
            decoder_mems_list,
            enc_states,
            enc_masks,
        )
        t = logits.argmax()
        if t == eos:
            break
        tokens.append(t)
    print("len(tokens)", len(tokens))
    print("tokens", tokens)

    text = "".join([id2token[i] for i in tokens])

    underline = "▁"
    #  underline = b"\xe2\x96\x81".decode()

    text = text.replace(underline, " ").strip()
    print("text:", text)


if __name__ == "__main__":
    main()
