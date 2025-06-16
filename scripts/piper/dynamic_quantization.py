#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse

import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxruntime.quantization import QuantType, quantize_dynamic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-fp16",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-int8",
        type=str,
        required=True,
    )
    return parser.parse_args()


# for op_block_list, see also
# https://github.com/microsoft/onnxruntime/blob/089c52e4522491312e6839af146a276f2351972e/onnxruntime/python/tools/transformers/float16.py#L115
#
# libc++abi: terminating with uncaught exception of type Ort::Exception:
# Type Error: Type (tensor(float16)) of output arg (/dp/RandomNormalLike_output_0)
# of node (/dp/RandomNormalLike) does not match expected type (tensor(float)).
#
# libc++abi: terminating with uncaught exception of type Ort::Exception:
# This is an invalid model. Type Error: Type 'tensor(float16)' of input
# parameter (/enc_p/encoder/attn_layers.0/Constant_84_output_0) of
# operator (Range) in node (/Range_1) is invalid.
def export_onnx_fp16(onnx_fp32_path, onnx_fp16_path):
    onnx_fp32_model = onnxmltools.utils.load_model(onnx_fp32_path)
    onnx_fp16_model = convert_float_to_float16(
        onnx_fp32_model,
        keep_io_types=True,
        op_block_list=[
            "RandomNormalLike",
            "Range",
        ],
    )
    onnxmltools.utils.save_model(onnx_fp16_model, onnx_fp16_path)


def main():
    args = get_args()
    print(args)

    in_filename = args.input
    output_fp16 = args.output_fp16
    output_int8 = args.output_int8

    quantize_dynamic(
        model_input=in_filename,
        model_output=output_int8,
        weight_type=QuantType.QUInt8,
    )

    export_onnx_fp16(in_filename, output_fp16)


if __name__ == "__main__":
    main()
