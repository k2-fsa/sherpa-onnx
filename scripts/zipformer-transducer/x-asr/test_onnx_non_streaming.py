#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse

# You can download test files used by this script from
# https://modelscope.cn/models/csukuangfj/2026-05-26/files
import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--use-int8",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--wav",
        type=str,
        required=True,
    )
    return parser.parse_args()


def show_sess(sess, hint):
    print(f"---{hint} input---")
    for i in sess.get_inputs():
        print(i)

    print(f"---{hint} output---")

    for i in sess.get_outputs():
        print(i)
    print("=" * 5)
    print()


def load_tokens(filename):
    ans = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            t, i = line.strip().split()
            ans[int(i)] = t
            pass

    return ans


def load_audio(filename: str):
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel

    if sample_rate != 16000:
        import librosa

        data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def load_model(use_int8):
    if use_int8:
        model = OnnxModel(
            encoder="./encoder-epoch-99-avg-1.int8.onnx",
            decoder="./decoder-epoch-99-avg-1.onnx",
            joiner="./joiner-epoch-99-avg-1.int8.onnx",
        )
    else:
        model = OnnxModel(
            encoder="./encoder-epoch-99-avg-1.onnx",
            decoder="./decoder-epoch-99-avg-1.onnx",
            joiner="./joiner-epoch-99-avg-1.onnx",
        )
    return model


def compute_feat(
    samples: np.ndarray,
    sample_rate: int,
    max_len: int = -1,
):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, samples.tolist())
    online_fbank.input_finished()

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )

    if max_len > 0:
        if features.shape[0] > max_len:
            features = features[:max_len]
        elif features.shape[0] < max_len:
            features = np.pad(
                features,
                ((0, max_len - features.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
    else:
        # Pad to a length T where (T-7)//2 is divisible by 8
        # This ensures all downsampling factors (1,2,4,8) work without remainder
        T = features.shape[0]
        after_embed = (T - 7) // 2
        remainder = after_embed % 8
        if remainder != 0:
            # Increase T so that (T-7)//2 is next multiple of 8
            new_after_embed = after_embed + (8 - remainder)
            new_T = new_after_embed * 2 + 7
            features = np.pad(
                features,
                ((0, new_T - T), (0, 0)),
                mode="constant",
                constant_values=0,
            )

    features = np.ascontiguousarray(features)

    assert features.data.contiguous is True
    assert features.dtype == np.float32, features.dtype

    return features


class OnnxModel:
    def __init__(self, encoder, decoder, joiner):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        session_opts.log_severity_level = 3  # ONLY ERROR

        self.session_opts = session_opts

        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        self.joiner = ort.InferenceSession(
            joiner,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        show_sess(self.encoder, "encoder")
        show_sess(self.decoder, "decoder")
        show_sess(self.joiner, "joiner")

        encoder_meta = self.encoder.get_modelmeta().custom_metadata_map

        model_type = encoder_meta["model_type"]
        assert model_type == "zipformer2", model_type

        self.context_size = self.decoder.get_inputs()[0].shape[1]

    def run_encoder(self, x):
        """
        Args: x: (1, T, C]
        """
        x_len = np.array([x.shape[1]], dtype=np.int32)
        out = self.encoder.run(
            [self.encoder.get_outputs()[0].name],
            {
                self.encoder.get_inputs()[0].name: x,
                self.encoder.get_inputs()[1].name: x_len,
            },
        )
        return out[0]

    def run_decoder(self, hyp):
        hyp = np.array([hyp], dtype=np.int32)
        out = self.decoder.run(
            [self.decoder.get_outputs()[0].name],
            {self.decoder.get_inputs()[0].name: hyp},
        )
        return out[0]

    def run_joiner(self, encoder_out, decoder_out):
        out = self.joiner.run(
            [self.joiner.get_outputs()[0].name],
            {
                self.joiner.get_inputs()[0].name: encoder_out,
                self.joiner.get_inputs()[1].name: decoder_out,
            },
        )
        return out[0]


def main():
    args = get_args()
    print(vars(args))

    samples, sample_rate = load_audio(args.wav)

    features = compute_feat(
        samples=samples,
        sample_rate=sample_rate,
    )
    print("features", features.shape)

    id2token = load_tokens("./tokens.txt")

    model = load_model(use_int8=args.use_int8)

    blank_id = 0

    hyp = [blank_id] * model.context_size
    decoder_out = model.run_decoder(hyp)

    x = features[None]
    encoder_out = model.run_encoder(x)
    num_frames = encoder_out.shape[1]
    for k in range(num_frames):
        cur_encoder_out = encoder_out[0, k : k + 1]
        joiner_out = model.run_joiner(cur_encoder_out, decoder_out)
        token_id = joiner_out.argmax()
        if token_id != blank_id:
            hyp.append(token_id)
            decoder_out = model.run_decoder(hyp[-model.context_size :])
    print(hyp)
    tokens = [id2token[i] for i in hyp[model.context_size :]]
    print(tokens)
    text = "".join(tokens)
    print(text)


if __name__ == "__main__":
    main()
