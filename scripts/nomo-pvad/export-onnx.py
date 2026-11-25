#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# Export the NOMO-PVAD target-speaker VAD model to ONNX and attach the
# meta data that sherpa-onnx reads at runtime.
#
# NOMO-PVAD (https://huggingface.co/TuyaNomo/nomo-pvad) is a streaming
# target-speaker detector: given an enrollment d-vector for a target speaker,
# it emits one probability every 160 ms saying whether *that* speaker is
# talking.
#
# The network takes
#   - spk_emb: (N, 192) enrollment embedding
#   - fbank:   (N, T, 80) 80-dim fbank, 25 ms window / 10 ms hop, dither=0,
#              with the time mean subtracted
# and returns
#   - chunk_logits: (N, N_chunks, 2); softmax(...)[..., 1] is P(target speaker)
#
# T must be a multiple of 16, since the ERes2NetV2 trunk downsamples time by 8
# and the chunk head averages every 2 trunk frames.
#
# Usage:
#   pip install torch onnx onnxruntime modelscope addict huggingface_hub
#   ./export-onnx.py --output ./nomo-pvad.onnx
#
# `addict` is not pulled in by `pip install modelscope`, but the ERes2NetV2
# definition that modelscope ships imports it.

import argparse
import os
import sys
from typing import Dict

import onnx
import torch


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="./nomo-pvad.onnx",
        help="Path to the exported ONNX model",
    )
    parser.add_argument(
        "--pvad-dir",
        type=str,
        default="",
        help="""Local copy of https://huggingface.co/TuyaNomo/nomo-pvad.
        It must contain core/, nomo_pvad.py and weights/nomo_pvad.pt.
        If empty, it is downloaded from Hugging Face.""",
    )
    parser.add_argument(
        "--eres2netv2-dir",
        type=str,
        default="",
        help="""Local directory containing pretrained_eres2netv2.ckpt.
        If empty, it is downloaded from ModelScope.""",
    )
    return parser.parse_args()


class PVADExport(torch.nn.Module):
    """ONNX-friendly wrapper around NomoPVADNet.

    NomoPVADNet.forward() also computes a chunk mask from the input lengths.
    The mask only depends on the input length, so we drop it here and let the
    caller work out how many chunks are valid.
    """

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, spk_emb: torch.Tensor, fbank: torch.Tensor) -> torch.Tensor:
        frame_feat = self.net.backbone(fbank)
        fused = self.net.film(frame_feat, spk_emb)
        x = self.net.feat_proj(fused)
        enc = self.net.speech_encoder(x)
        det = self.net.det_decoder(enc)
        return self.net.classifier_det(self.net._extract_chunks(det))


def main():
    args = get_args()

    pvad_dir = args.pvad_dir
    if not pvad_dir:
        from huggingface_hub import snapshot_download

        pvad_dir = snapshot_download("TuyaNomo/nomo-pvad")

    eres2netv2_dir = args.eres2netv2_dir
    if not eres2netv2_dir:
        from modelscope import snapshot_download as ms_download

        eres2netv2_dir = ms_download("iic/speech_eres2netv2_sv_zh-cn_16k-common")

    if pvad_dir not in sys.path:
        sys.path.insert(0, pvad_dir)

    from core.model import NomoPVADNet  # noqa: E402  (comes from pvad_dir)

    ckpt = os.path.join(pvad_dir, "weights", "nomo_pvad.pt")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)

    # Only the architecture fields are read from the checkpoint; the backbone
    # directory is injected by us.
    cfg = dict(state["model_cfg"])
    cfg["eres2netv2_model_dir"] = eres2netv2_dir

    net = NomoPVADNet(cfg)
    net.load_state_dict(state["model_state_dict"], strict=False)
    net.eval()
    # The released checkpoint is fp16. Export as fp32: fp16 CPU inference in
    # onnxruntime is both slower and less accurate.
    net.float()

    n_params = sum(p.numel() for p in net.parameters())
    print(f"nomo-pvad: {n_params / 1e6:.2f}M parameters")

    pvad = PVADExport(net).eval()

    spk_emb = torch.nn.functional.normalize(torch.randn(1, 192), dim=-1)
    fbank = torch.randn(1, 240, 80)  # 2.4 s of 10 ms frames

    torch.onnx.export(
        pvad,
        (spk_emb, fbank),
        args.output,
        input_names=["spk_emb", "fbank"],
        output_names=["chunk_logits"],
        dynamic_axes={
            "spk_emb": {0: "N"},
            "fbank": {0: "N", 1: "T"},
            "chunk_logits": {0: "N", 1: "num_chunks"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    meta_data = {
        "model_type": "nomo-pvad",
        "version": "1",
        "sample_rate": 16000,
        # The model emits one probability per 160 ms chunk.
        "chunk_samples": 2560,
        # 2.4 s of audio is re-scored for every chunk. This is the sliding
        # window used by the reference implementation.
        "window_samples": 38400,
        "feature_dim": 80,
        "spk_embed_dim": 192,
        "frame_length_ms": 25,
        "frame_shift_ms": 10,
        "normalize_samples": 1,
        "feature_normalize_type": "global-mean",
        "url": "https://huggingface.co/TuyaNomo/nomo-pvad",
        "comment": "Target-speaker VAD. Apache-2.0, based on ERes2NetV2.",
    }
    add_meta_data(args.output, meta_data)

    print(f"Saved to {args.output}")
    print(f"Size: {os.path.getsize(args.output) / 1e6:.1f} MB")

    # Sanity check: ONNX Runtime must agree with PyTorch.
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])

    for T in (32, 240, 400):
        fb = torch.randn(1, T, 80)
        with torch.no_grad():
            ref = pvad(spk_emb, fb).numpy()
        got = sess.run(
            None,
            {"spk_emb": spk_emb.numpy(), "fbank": fb.numpy()},
        )[0]
        assert got.shape == ref.shape, (got.shape, ref.shape)
        diff = np.abs(ref - got).max()
        print(f"  T={T:4d} num_chunks={ref.shape[1]:3d} max_abs_diff={diff:.3e}")
        assert diff < 1e-4, diff

    print("Verified: ONNX matches PyTorch")


if __name__ == "__main__":
    main()
