#!/usr/bin/env python3
"""Add sherpa-onnx metadata (and the iSTFT notch filter) to an exported 7M model.

Two things here are specific to these distills and were found the hard way:

1. iSTFT image tones.
   The ISTFTNet decoder leaves narrow image tones at 4.8 kHz and 9.6 kHz
   (sr/5 and 2*sr/5 for the 20-point inverse STFT). A 65-tap FIR notch is baked
   into the graph as a Conv between Slice_3 and Squeeze_3 so every runtime gets
   filtered audio without extra post-processing. Measured on the sample
   sentences, the 4.8 kHz / 9.6 kHz energy ratio drops from 1.46 / 4.71 to
   0.10-0.43.

2. Node insertion order.
   ONNX requires nodes in topological order. Appending the Conv at the end of
   graph.node produces a model that onnx.checker rejects, so the new node is
   inserted at the consumer's index instead.

Usage:
    python add_meta_data.py --model model.fp32.onnx --lang ar \
        --comment "Nabra-7M-Distill Arabic FP32"
"""

import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper

SAMPLE_RATE = 24000
# 20-point inverse STFT -> image tones at sr/5 and 2*sr/5.
NOTCH_FREQS = (4800.0, 9600.0)
NOTCH_TAPS = 65
NOTCH_BW = 300.0
NOTCH_DEPTH = 6


def design_notch(freqs=NOTCH_FREQS, taps=NOTCH_TAPS, sr=SAMPLE_RATE,
                 bw=NOTCH_BW, depth=NOTCH_DEPTH):
    """Windowed-sinc band-stop cascade, flattened to a single FIR kernel."""
    n = np.arange(taps) - (taps - 1) / 2
    kernel = np.zeros(taps)
    kernel[(taps - 1) // 2] = 1.0
    for _ in range(depth):
        for f0 in freqs:
            lo, hi = (f0 - bw / 2) / sr, (f0 + bw / 2) / sr
            band = (2 * hi * np.sinc(2 * hi * n)) - (2 * lo * np.sinc(2 * lo * n))
            band *= np.hamming(taps)
            stop = -band
            stop[(taps - 1) // 2] += 1.0
            kernel = np.convolve(kernel, stop, mode="same")
    return (kernel / kernel.sum()).astype(np.float32)


NOTCH_CONV_NAME = "notch_fir"
NOTCH_WEIGHT_NAME = "notch_fir_w"
NOTCH_OUTPUT_NAME = "audio_notched"


def _notch_markers(graph) -> dict:
    """Which pieces of a previous notch insertion are present in the graph."""
    return {
        "conv": any(n.name == NOTCH_CONV_NAME for n in graph.node),
        "weight": any(i.name == NOTCH_WEIGHT_NAME for i in graph.initializer),
        "rewired": any(NOTCH_OUTPUT_NAME in n.input for n in graph.node),
    }


def bake_notch(model: onnx.ModelProto) -> bool:
    """Insert the FIR notch immediately before the final Squeeze.

    The exported graph ends Slice_3 -> Squeeze_3 -> ... -> audio, with an If
    node in between (dynamic-shape guard), so the Squeeze is located by name
    rather than by being the producer of the graph output.

    Idempotent: a model that already carries the notch is returned untouched.
    Re-running the insertion would otherwise wire the Conv's output back into
    its own input and duplicate the initializer, producing a graph that
    onnx.checker rejects.
    """
    graph = model.graph

    markers = _notch_markers(graph)
    if all(markers.values()):
        return True
    if any(markers.values()):
        raise RuntimeError(
            f"graph carries a partial notch insertion ({markers}); refusing to "
            "mutate it. Re-export the model from the checkpoint."
        )

    squeeze_idx = squeeze = None
    for i, node in enumerate(graph.node):
        if node.op_type == "Squeeze" and node.name.endswith("Squeeze_3"):
            squeeze_idx, squeeze = i, node
            break
    if squeeze is None:
        return False

    src = squeeze.input[0]
    kernel = design_notch().reshape(1, 1, -1)
    graph.initializer.append(numpy_helper.from_array(kernel, NOTCH_WEIGHT_NAME))
    conv = helper.make_node(
        "Conv", inputs=[src, NOTCH_WEIGHT_NAME], outputs=[NOTCH_OUTPUT_NAME],
        name=NOTCH_CONV_NAME, kernel_shape=[NOTCH_TAPS],
        pads=[(NOTCH_TAPS - 1) // 2, (NOTCH_TAPS - 1) // 2], group=1,
    )
    # ONNX requires topological order: insert at the consumer's index rather
    # than appending, otherwise onnx.checker.check_model rejects the graph.
    graph.node.insert(squeeze_idx, conv)
    squeeze.input[0] = NOTCH_OUTPUT_NAME
    return True


def stamp(model: onnx.ModelProto, meta: dict) -> None:
    model.ClearField("metadata_props")
    for key, value in meta.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, str(value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lang", required=True,
                   help="espeak-ng voice, e.g. 'ar' or 'en-us'")
    p.add_argument("--language", default="", help="human-readable language name")
    p.add_argument("--comment", default="")
    p.add_argument("--skip-notch", action="store_true")
    args = p.parse_args()

    model = onnx.load(args.model)
    if not args.skip_notch:
        if not bake_notch(model):
            raise SystemExit(
                "could not locate Squeeze_3: the FIR notch was NOT inserted. "
                "Refusing to write an unfiltered model — re-export first, or "
                "pass --skip-notch if that is genuinely intended."
            )
        print("notch baked: True")

    stamp(model, {
        "model_type": "kokoro",
        "sample_rate": SAMPLE_RATE,
        "n_speakers": 1,
        "style_dim": "510,1,256",
        "has_espeak": 1,
        "max_token_len": 510,
        "version": 1,
        "language": args.language or args.lang,
        "voice": args.lang,
        "id2speaker": "0->af_msa",
        "speaker2id": "af_msa->0",
        "speaker_names": "af_msa",
        "comment": args.comment,
    })
    # Validate before overwriting the input: a failed check must not leave a
    # corrupt file where a working model used to be.
    onnx.checker.check_model(model)
    onnx.save(model, args.model)
    print("checked + saved:", args.model)


if __name__ == "__main__":
    main()
