#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse

import nemo.collections.asr as nemo_asr
import torch


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--max-len",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--model-id",
        type=str,
        help="e.g., nvidia/parakeet-tdt_ctc-110m",
        required=True,
    )
    return parser.parse_args()


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        """
        x: feature tensor, (N, C, T)
        """
        x_lens = torch.tensor([x.shape[2]], dtype=torch.int64)
        encoder_output, encoder_out_lens = self.model.encoder(
            audio_signal=x, length=x_lens
        )
        log_probs = self.model.ctc_decoder(encoder_output=encoder_output)

        return log_probs


@torch.no_grad()
def main():
    args = get_args()
    print(vars(args))

    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name=args.model_id
    )
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model_id)
    asr_model.change_decoding_strategy(decoder_type="ctc")

    asr_model.eval()
    print(type(asr_model))
    print(asr_model)
    print(asr_model.cfg)

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")

    m = ModelWrapper(asr_model)
    m.eval()
    feat_dim = asr_model.cfg["preprocessor"]["features"]
    print("feat_dim", feat_dim)
    x = torch.rand(1, feat_dim, args.max_len, dtype=torch.float32)

    torch.onnx.export(
        m,
        x,
        "model.onnx",
        opset_version=13,
        input_names=["x"],
        output_names=["log_probs"],
    )


if __name__ == "__main__":
    main()
