#!/usr/bin/env python3
"""
Make a multilingual Pocket TTS bundle of
https://huggingface.co/KevinAHM/pocket-tts-onnx usable with sherpa-onnx.

The multilingual bundles (german, french_24l, spanish, italian, portuguese, ...)
set "insert_bos_before_voice": true in their bundle.json: a learned embedding
(bos_before_voice.npy) has to go in front of the voice embedding that the
mimi encoder computes from the reference audio. Without it, the model finds
no end of the sentence and produces noise.

This script appends that embedding inside the encoder graph, so the encoder
output already starts with it and the C++ code needs no change:

    python3 ./add_bos_before_voice.py \
      --encoder ./mimi_encoder_int8.onnx \
      --bos ./bos_before_voice.npy \
      --output ./encoder.onnx

Requires: onnx, numpy
"""

import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, help="mimi encoder of the bundle")
    parser.add_argument("--bos", required=True, help="bos_before_voice.npy of the bundle")
    parser.add_argument("--output", required=True, help="where to write the new encoder")
    return parser.parse_args()


def main():
    args = get_args()
    model = onnx.load(args.encoder)
    bos = np.load(args.bos).astype(np.float32)

    graph = model.graph
    output = graph.output[0]
    # (batch, frames, dim) for the encoder output and (1, 1, dim) for bos
    assert bos.ndim == 3 and bos.shape[:2] == (1, 1), bos.shape
    dims = output.type.tensor_type.shape.dim
    assert len(dims) == 3, f"expected a 3-D encoder output, got {len(dims)} dims"
    if dims[2].HasField("dim_value") and dims[2].dim_value != bos.shape[2]:
        raise ValueError(
            f"bos has width {bos.shape[2]}, the encoder output {dims[2].dim_value}"
        )

    # Everything that used the encoder output so far keeps using it without bos;
    # only the graph output gets the Concat.
    without_bos = output.name + "_without_bos"
    for node in graph.node:
        node.output[:] = [without_bos if o == output.name else o for o in node.output]
        node.input[:] = [without_bos if i == output.name else i for i in node.input]

    graph.initializer.append(numpy_helper.from_array(bos, name="bos_before_voice"))
    graph.node.append(
        helper.make_node("Concat", ["bos_before_voice", without_bos], [output.name], axis=1)
    )
    dims[1].dim_param = "frames_with_bos"

    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
