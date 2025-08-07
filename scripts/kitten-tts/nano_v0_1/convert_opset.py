#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
Change the model so that it can be run in onnxruntime 1.17.1
"""

import onnx


def main():
    model = onnx.load("kitten_tts_nano_v0_1.onnx")

    # Print current opsets
    for opset in model.opset_import:
        print(f"Domain: '{opset.domain}', Version: {opset.version}")

    # Modify the opset versions (be careful!)
    for opset in model.opset_import:
        if opset.domain == "":  # ai.onnx domain
            opset.version = 19  # change from 20 to 19
        elif opset.domain == "ai.onnx.ml":
            opset.version = 4  # change from 5 to 4

    # Save the modified model
    onnx.save(model, "kitten_tts_nano_v0_1_patched.onnx")


if __name__ == "__main__":
    main()
