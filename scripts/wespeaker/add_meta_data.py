#!/usr/bin/env python3
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This script adds meta data to a model so that it can be used in sherpa-onnx.

Usage:
./add_meta_data.py --model ./voxceleb_resnet34.onnx  --language English
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict

import onnx
import onnxruntime
from onnx import numpy_helper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the input onnx model. Example value: model.onnx",
    )

    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="""Supported language of the input model.
        Example value: Chinese, English.
        """,
    )

    parser.add_argument(
        "--url",
        type=str,
        default="https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md",
        help="Where the model is downloaded",
    )

    parser.add_argument(
        "--comment",
        type=str,
        default="no comment",
        help="Comment about the model",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate expected by the model",
    )

    return parser.parse_args()


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


def remove_no_op_pad_before_average_pool(filename: str) -> int:
    """Remove Pad nodes that pad nothing and feed only AveragePool nodes.
    The model is changed in-place, and it is saved only if a node is removed.

    The CAM++ models from wespeaker contain, for each segment pooling,
    Pad(pads=0) followed by AveragePool(ceil_mode=1). The Pad does not
    change its input, but onnxruntime 1.21 and later fuse it into the
    AveragePool and set count_include_pad=1 there. For the last, partial
    window, onnxruntime 1.28 and older then divide the sum by the kernel
    size instead of by the number of frames in that window, so with
    onnxruntime 1.21 to 1.28 the embedding depends on the input length
    modulo 2 seconds.

    Without the Pad, old and new versions of onnxruntime compute the same
    result as PyTorch. Models without such a Pad, e.g., the ResNet models,
    are not changed.

    See https://github.com/k2-fsa/sherpa-onnx/issues/4201

    Args:
      filename:
        Filename of the ONNX model to be changed.
    Returns:
      Return the number of removed Pad nodes.
    """
    model = onnx.load(filename)
    graph = model.graph

    constants = {t.name: t for t in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant" and node.attribute[0].name == "value":
            constants[node.output[0]] = node.attribute[0].t

    consumers = defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node)

    graph_outputs = {o.name for o in graph.output}

    removed = []
    for node in graph.node:
        if node.op_type != "Pad" or len(node.input) < 2:
            continue

        pads = constants.get(node.input[1])
        if pads is None or numpy_helper.to_array(pads).any():
            continue

        users = consumers[node.output[0]]
        if node.output[0] in graph_outputs or not users:
            continue

        if any(u.op_type != "AveragePool" for u in users):
            continue

        for u in users:
            for i, name in enumerate(u.input):
                if name == node.output[0]:
                    u.input[i] = node.input[0]

        removed.append(node)

    if not removed:
        return 0

    for node in removed:
        graph.node.remove(node)

    # Remove the pads constants that are no longer used
    used = {name for node in graph.node for name in node.input}
    unused = {node.input[1] for node in removed} - used
    for node in [n for n in graph.node if n.op_type == "Constant"]:
        if node.output[0] in unused:
            graph.node.remove(node)

    for t in [t for t in graph.initializer if t.name in unused]:
        graph.initializer.remove(t)

    onnx.save(model, filename)

    return len(removed)


def get_output_dim(filename) -> int:
    filename = str(filename)
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 3  # error level
    sess = onnxruntime.InferenceSession(filename, session_opts)

    for i in sess.get_inputs():
        print(i)

    print("----------")

    for o in sess.get_outputs():
        print(o)

    print("----------")

    assert len(sess.get_inputs()) == 1
    assert len(sess.get_outputs()) == 1

    i = sess.get_inputs()[0]
    o = sess.get_outputs()[0]

    assert i.shape[:2] == ["B", "T"], i.shape
    assert o.shape[0] == "B"

    assert i.shape[2] == 80, i.shape

    return o.shape[1]


def main():
    args = get_args()
    model = Path(args.model)
    language = args.language
    url = args.url
    comment = args.comment
    sample_rate = args.sample_rate

    if not model.is_file():
        raise ValueError(f"{model} does not exist")

    assert len(language) > 0, len(language)
    assert len(url) > 0, len(url)

    num_removed = remove_no_op_pad_before_average_pool(str(model))
    print(f"Removed {num_removed} no-op Pad node(s) before AveragePool")

    output_dim = get_output_dim(model)

    # all models from wespeaker expect input samples in the range
    # [-32768, 32767]
    normalize_samples = 0

    meta_data = {
        "framework": "wespeaker",
        "language": language,
        "url": url,
        "comment": comment,
        "sample_rate": sample_rate,
        "output_dim": output_dim,
        "normalize_samples": normalize_samples,
    }
    print(meta_data)
    add_meta_data(filename=str(model), meta_data=meta_data)


if __name__ == "__main__":
    main()
