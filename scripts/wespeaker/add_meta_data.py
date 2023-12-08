#!/usr/bin/env python3
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This script adds meta data to a model so that it can be used in sherpa-onnx.

Usage:
./add_meta_data.py --model ./voxceleb_resnet34.onnx  --language English
"""

import argparse
from pathlib import Path
from typing import Dict

import onnx
import onnxruntime


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

    output_dim = get_output_dim(model)

    # all models from wespeaker expect input samples in the range
    # [-32768, 32767]
    normalize_features = 0

    meta_data = {
        "framework": "wespeaker",
        "language": language,
        "url": url,
        "comment": comment,
        "sample_rate": sample_rate,
        "output_dim": output_dim,
        "normalize_features": normalize_features,
    }
    print(meta_data)
    add_meta_data(filename=str(model), meta_data=meta_data)


if __name__ == "__main__":
    main()
