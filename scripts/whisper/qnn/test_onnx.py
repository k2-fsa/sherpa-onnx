#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import whisper

from test_torch import compute_feat
from export_onnx import causal_mask_1d


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
        for i in range(4):
            k = np.zeros((batch_size, 448, 384), dtype=np.float32)
            v = np.zeros((batch_size, 448, 384), dtype=np.float32)
            self_cache.extend([k, v])
        return self_cache


def main():
    torch_model = whisper.load_model("tiny.en")
    tokenizer = whisper.tokenizer.get_tokenizer(
        torch_model.is_multilingual, num_languages=torch_model.num_languages
    )
    print(tokenizer.sot)  # 50527

    mel = compute_feat("./en.wav").numpy()
    print(mel.shape)  # (1, 80. 3000)
    model = OnnxModel("./tiny.en-encoder.onnx", "./tiny.en-decoder.onnx")

    cross_kv = model.run_encoder(mel)
    print(len(cross_kv))  # 8

    self_kv = model.get_self_cache()

    eot = 50256
    token = np.array([[50257]], dtype=np.int32)  # sot
    offset = np.array([0], dtype=np.int32)
    mask = causal_mask_1d(offset.item(), 448).numpy()

    out = model.run_decoder([token] + self_kv + cross_kv + [offset, mask])

    logits = out[0]

    torch.save(torch.from_numpy(logits), "logits_onnx.pt")

    for i in range(1, 9):
        self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]

    print(logits[0, 0].argmax())

    token = np.array([[50362]], dtype=np.int32)  # no_timestamps
    offset += 1
    mask = causal_mask_1d(offset.item(), 448).numpy()

    out = model.run_decoder([token] + self_kv + cross_kv + [offset, mask])
    idx = out[0][0, 0].argmax()

    eot = 50256
    t = 0

    ans = []

    while idx != eot and t < 100:
        t += 1

        ans.append(idx)
        token = np.array([[idx]], dtype=np.int32)  # no_timestamps
        for i in range(1, 9):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]

        offset += 1
        mask = causal_mask_1d(offset.item(), 448).numpy()

        out = model.run_decoder([token] + self_kv + cross_kv + [offset, mask])
        idx = out[0][0, 0].argmax()

    print(ans)
    text = "".join(tokenizer.decode(ans)).strip()
    print(text)


if __name__ == "__main__":
    main()
