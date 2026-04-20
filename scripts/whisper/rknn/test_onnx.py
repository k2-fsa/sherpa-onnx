#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import whisper

from test_torch import compute_feat
from export_onnx import causal_mask_1d, get_args


class OnnxModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.init_encoder(encoder)
        self.init_decoder(decoder)

    def init_encoder(self, encoder: str):
        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        self.encoder_input_names = []
        self.encoder_output_names = []

        print(f"-----{encoder}-----")
        print(f"----input----")
        for i in self.encoder.get_inputs():
            print(i)
            self.encoder_input_names.append(i.name)

        print("-----output-----")

        for i in self.encoder.get_outputs():
            print(i)
            self.encoder_output_names.append(i.name)

        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.n_text_layer = int(meta["n_text_layer"])
        self.n_text_ctx = int(meta["n_text_ctx"])
        self.n_text_state = int(meta["n_text_state"])

    def init_decoder(self, decoder: str):
        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        self.decoder_input_names = []
        self.decoder_output_names = []

        print(f"-----{decoder}-----")
        print(f"----input----")
        for i in self.decoder.get_inputs():
            print(i)
            self.decoder_input_names.append(i.name)

        print("-----output-----")

        for i in self.decoder.get_outputs():
            print(i)
            self.decoder_output_names.append(i.name)

    def run_encoder(
        self,
        mel: np.ndarray,
    ) -> List[np.ndarray]:
        cross_kv = self.encoder.run(
            self.encoder_output_names,
            {
                self.encoder.get_inputs()[0].name: mel,
            },
        )
        return cross_kv

    def run_decoder(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        feed = {
            self.decoder.get_inputs()[i].name: inputs[i] for i in range(len(inputs))
        }

        out = self.decoder.run(
            self.decoder_output_names,
            feed,
        )
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


def main():
    args = get_args()
    print(vars(args))

    torch_model = whisper.load_model(args.model)
    tokenizer = whisper.tokenizer.get_tokenizer(
        torch_model.is_multilingual, num_languages=torch_model.num_languages
    )

    mel = compute_feat("./en-16k.wav").numpy()
    print(mel.shape)  # (1, 80. 3000)
    model = OnnxModel(f"./{args.model}-encoder.onnx", f"./{args.model}-decoder.onnx")

    sot_sequence = list(tokenizer.sot_sequence) + [tokenizer.no_timestamps]

    # tiny.en: [50257, 50362]
    # tiny: [50258, 50259, 50359, 50363]
    print("sot sequence", sot_sequence)

    cross_kv = model.run_encoder(mel)
    print(len(cross_kv))  # 8

    self_kv = model.get_self_cache()

    # tiny.en: 50256
    # tiny: 50257
    eot = tokenizer.eot
    print("eot", eot)

    offset = np.array([0], dtype=np.int32)
    for t in sot_sequence:
        token = np.array([[t]], dtype=np.int32)  # sot
        mask = causal_mask_1d(offset.item(), model.n_text_ctx).numpy()

        out = model.run_decoder([token] + self_kv + cross_kv + [offset, mask])

        for i in range(1, len(out)):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]

        offset += 1

    idx = out[0][0, 0].argmax()

    ans = []

    while idx != eot and offset.item() < 200:
        ans.append(idx)
        token = np.array([[idx]], dtype=np.int32)  # no_timestamps
        for i in range(1, len(out)):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]

        mask = causal_mask_1d(offset.item(), model.n_text_ctx).numpy()

        out = model.run_decoder([token] + self_kv + cross_kv + [offset, mask])
        idx = out[0][0, 0].argmax()

        offset += 1

    print(ans)
    text = "".join(tokenizer.decode(ans)).strip()
    print(text)


if __name__ == "__main__":
    main()
