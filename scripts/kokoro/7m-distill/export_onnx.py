"""Export oddadmix's 7M distilled Kokoro models (Nabra Arabic, Kokoro English) to ONNX
with the exact I/O contract sherpa-onnx's Kokoro runtime expects.

Contract (must match the already-published 82M artefacts):
    input_ids  INT64   [batch_size, sequence_length]
    ref_s      FLOAT   [batch_size, 256]
    speed      FLOAT   [1]
  -> audio      FLOAT   [sequence_length * 300]   (1-D, batch squeezed)

Usage:
    python scripts/export_7m_onnx.py --repo oddadmix/Nabra-7M-Distill \
        --dir nabra_7m --weights kokoro_arabic_7m.pth --out models/nabra7m_fp32.onnx
"""
import argparse
import os
import sys

import numpy as np
import torch

# Minimum spectral cosine between the torch reference and the exported graph.
# Measured on these models: two torch runs of the same input score 0.89-0.94
# (SineGen randomness), a correct export scores the same, and unrelated noise
# scores 0.23. 0.70 sits clear of both.
SPECTRAL_MIN = 0.70


class SherpaWrapper(torch.nn.Module):
    """KModel -> (waveform, duration) becomes just the squeezed waveform."""

    def __init__(self, kmodel):
        super().__init__()
        self.kmodel = kmodel

    def forward(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor,
                speed: torch.FloatTensor):
        waveform, _duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform.squeeze(0)


def load_kmodel(repo_dir: str, weights: str, config: str, repo_id: str):
    """Load through the vendored patched kokoro (upstream hardcodes 1024/512 widths)."""
    here = os.path.abspath(repo_dir)
    sys.path.insert(0, os.path.join(here, "kokoro_patched"))
    sys.path.insert(0, here)
    from kokoro import KModel  # noqa: E402

    m = KModel(
        repo_id=repo_id,
        config=os.path.join(here, config),
        model=os.path.join(here, weights),
        disable_complex=True,
    ).eval()
    try:
        from arabic_g2p import EXTRA_SYMBOLS
        m.vocab.update(EXTRA_SYMBOLS)
    except ImportError:
        pass  # English package has no arabic_g2p
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id, for KModel's config fetch")
    ap.add_argument("--dir", required=True, help="local dir with the cloned model files")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    m = load_kmodel(args.dir, args.weights, args.config, args.repo)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params:,}")

    wrapper = SherpaWrapper(m)

    # Probe shapes with a realistic sentence so the tracer sees a non-trivial graph.
    input_ids = torch.LongTensor([[0] + list(range(1, 41)) + [0]])
    ref_s = torch.randn(1, 256)
    speed = torch.tensor([1.0], dtype=torch.float32)

    with torch.no_grad():
        ref_audio, _ = m.forward_with_tokens(input_ids, ref_s, speed)
    print(f"torch reference audio: {tuple(ref_audio.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args=(input_ids, ref_s, speed),
            f=args.out,
            export_params=True,
            verbose=False,
            input_names=["input_ids", "ref_s", "speed"],
            output_names=["audio"],
            opset_version=args.opset,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "ref_s": {0: "batch_size"},
                "audio": {0: "sequence_length"},
            },
            do_constant_folding=True,
        )

    # --- structural check -------------------------------------------------
    import onnx
    from onnx import TensorProto

    model = onnx.load(args.out, load_external_data=False)
    onnx.checker.check_model(args.out)          # raises on malformed graph
    print("onnx.checker: PASS")

    def sig(items):
        return [
            (t.name, TensorProto.DataType.Name(t.type.tensor_type.elem_type),
             [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim])
            for t in items
        ]

    print("inputs :", sig(model.graph.input))
    print("outputs:", sig(model.graph.output))

    # --- numerical check against torch ------------------------------------
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {
        "input_ids": input_ids.numpy(),
        "ref_s": ref_s.numpy(),
        "speed": speed.numpy(),
    })
    audio = outs[0]
    print(f"onnx audio: {audio.shape}")

    ref = ref_audio.squeeze(0).numpy()
    n = min(len(ref), len(audio))
    if n == 0:
        raise SystemExit("export produced empty audio; refusing to ship it")
    if len(ref) != len(audio):
        raise SystemExit(
            f"length mismatch: torch {len(ref)} vs onnx {len(audio)}. The "
            "export does not reproduce the reference pipeline."
        )

    # A sample-wise tolerance is not meaningful for this decoder: SineGen draws
    # a fresh random excitation per run, so two *torch* runs of the same input
    # differ by more than torch differs from ONNX (measured: 1.59 vs 1.48
    # relative). Compare long-term spectra instead, which are stable under that
    # randomness and still catch a genuinely broken export.
    def spectrum(x):
        s = np.abs(np.fft.rfft(x))
        return s / (s.sum() + 1e-12)

    p, q = spectrum(ref[:n]), spectrum(audio[:n])
    cos = float(p @ q / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-12))
    print(f"spectral cosine torch/onnx = {cos:.4f}")
    if cos < SPECTRAL_MIN:
        raise SystemExit(
            f"spectral check failed: cosine {cos:.4f} < {SPECTRAL_MIN}. The "
            "ONNX graph does not match the torch pipeline."
        )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  {size_mb:.1f} MB")


if __name__ == "__main__":
    main()