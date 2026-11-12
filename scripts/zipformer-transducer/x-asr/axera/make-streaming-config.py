#!/usr/bin/env python3
"""Generate the Pulsar2 config for a streaming Zipformer encoder.

A streaming encoder has one calibration tar per input (``x`` plus every cache),
so the 100+ entry config is generated here instead of being checked in.

Usage::

    python3 make-streaming-config.py \\
        --onnx encoder-streaming-ax650.onnx \\
        --calib-dir ./stream_calib \\
        --output ./pulsar2-streaming-encoder.json \\
        --output-dir ./output_streaming --work-dir ./output_streaming/work
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx

DTYPES = {1: "FP32", 2: "U8", 3: "S8", 4: "U16", 5: "S16", 6: "U32", 7: "S64", 10: "FP16"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib-dir", required=True, help="directory written by make-streaming-calibration-set.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--calibration-size", type=int, default=16)
    ap.add_argument("--output-name", default="encoder_streaming.axmodel")
    ap.add_argument("--output-dir", default="./output_streaming")
    ap.add_argument("--work-dir", default="./output_streaming/work")
    args = ap.parse_args()

    model = onnx.load(args.onnx, load_external_data=False)
    inputs = []
    for i in model.graph.input:
        dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
        inputs.append((i.name, i.type.tensor_type.elem_type, dims))

    calib = Path(args.calib_dir)
    input_configs = []
    for name, _etype, _dims in inputs:
        tar = calib / f"{name}.tar.gz"
        if not tar.exists():
            raise SystemExit(f"missing calibration tar: {tar}")
        input_configs.append({
            "tensor_name": name,
            "calibration_dataset": str(tar),
            "calibration_format": "Numpy",
            "calibration_size": args.calibration_size,
            "calibration_mean": [],
            "calibration_std": [],
        })

    # every input except `x` is a cache; they are plain float tensors
    processors = [{"tensor_name": "DEFAULT", "tensor_layout": "NCHW",
                   "src_dtype": "FP32", "src_layout": "NCHW"}]
    for name, etype, _dims in inputs:
        if DTYPES.get(etype, "FP32") != "FP32":
            processors.append({"tensor_name": name, "tensor_layout": "NCHW",
                               "src_dtype": DTYPES[etype], "src_layout": "NCHW"})

    cfg = {
        "model_type": "ONNX",
        "target_hardware": "AX650",
        "npu_mode": "NPU1",
        "input": args.onnx,
        "output_dir": args.output_dir,
        "output_name": args.output_name,
        "work_dir": args.work_dir,
        "input_shapes": ",".join(f"{n}:{'x'.join(str(v) for v in d)}" for n, _t, d in inputs),
        "onnx_opt": {"disable_onnx_optimization": False, "enable_onnxsim": False,
                     "model_check": True},
        "quant": {
            "input_configs": input_configs,
            "calibration_method": "MinMax",
            "precision_analysis": False,
            "highest_mix_precision": False,
            "conv_bias_data_type": "FP32",
            "disable_auto_refine_scale": True,
        },
        "input_processors": processors,
        "compiler": {"check": 0},
    }
    Path(args.output).write_text(json.dumps(cfg, indent=1))
    print(f"{len(inputs)} inputs -> {args.output}")


if __name__ == "__main__":
    main()
