#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

import argparse
import logging
from pathlib import Path

from rknn.api import RKNN

logging.basicConfig(level=logging.WARNING)

g_platforms = [
    #  "rk3562",
    #  "rk3566",
    #  "rk3568",
    #  "rk3576",
    "rk3588",
]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--target-platform",
        type=str,
        required=True,
        help=f"Supported values are: {','.join(g_platforms)}",
    )

    parser.add_argument(
        "--in-model",
        type=str,
        required=True,
        help="Path to the input onnx model",
    )

    parser.add_argument(
        "--out-model",
        type=str,
        required=True,
        help="Path to the output rknn model",
    )

    return parser


def get_meta_data(model: str):
    import onnxruntime

    session_opts = onnxruntime.SessionOptions()
    session_opts.inter_op_num_threads = 1
    session_opts.intra_op_num_threads = 1

    m = onnxruntime.InferenceSession(
        model,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    for i in m.get_inputs():
        print(i)

    print("-----")

    for i in m.get_outputs():
        print(i)
    print()

    meta = m.get_modelmeta().custom_metadata_map
    s = ""
    sep = ""
    for key, value in meta.items():
        s = s + sep + f"{key}={value}"
        sep = ";"
    assert len(s) < 1024, len(s)

    print("len(s)", len(s), s)

    return s


def export_rknn(rknn, filename):
    ret = rknn.export_rknn(filename)
    if ret != 0:
        exit(f"Export rknn model to {filename} failed!")


def init_model(filename: str, target_platform: str, custom_string=None):
    rknn = RKNN(verbose=False)

    rknn.config(
        optimization_level=0,
        target_platform=target_platform,
        custom_string=custom_string,
    )
    if not Path(filename).is_file():
        exit(f"{filename} does not exist")

    ret = rknn.load_onnx(model=filename)
    if ret != 0:
        exit(f"Load model {filename} failed!")

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        exit(f"Build model {filename} failed!")

    return rknn


class RKNNModel:
    def __init__(
        self,
        model: str,
        target_platform: str,
    ):
        meta = get_meta_data(model)
        print(meta)

        self.model = init_model(
            model,
            target_platform=target_platform,
            custom_string=meta,
        )

    def export_rknn(self, model):
        export_rknn(self.model, model)

    def release(self):
        self.model.release()


def main():
    args = get_parser().parse_args()
    print(vars(args))

    model = RKNNModel(
        model=args.in_model,
        target_platform=args.target_platform,
    )

    model.export_rknn(
        model=args.out_model,
    )

    model.release()


if __name__ == "__main__":
    main()
