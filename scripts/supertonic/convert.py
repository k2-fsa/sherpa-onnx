#!/usr/bin/env python3
# Copyright (c)  2026 zengyw

"""
Quantize Supertonic TTS ONNX models (duration_predictor, text_encoder,
vector_estimator, vocoder) to int8.
See also https://github.com/supertone-inc/supertonic
"""

import argparse
import glob
import inspect
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

try:
    from onnxruntime.quantization import CalibrationMethod
except Exception:
    CalibrationMethod = None

_quant_pre_process = None
try:
    from onnxruntime.quantization.shape_inference import quant_pre_process as _qpp
    _quant_pre_process = _qpp
except Exception:
    try:
        from onnxruntime.quantization import quant_pre_process as _qpp
        _quant_pre_process = _qpp
    except Exception:
        _quant_pre_process = None


def ensure_graph_names(m: onnx.ModelProto) -> None:
    def fix_graph(g: onnx.GraphProto, prefix: str) -> None:
        if not g.name:
            g.name = prefix
        for node in g.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH and attr.g is not None:
                    fix_graph(attr.g, f"{prefix}_{node.name or node.op_type}_g")
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for i, sg in enumerate(attr.graphs):
                        fix_graph(sg, f"{prefix}_{node.name or node.op_type}_gs{i}")

    if not m.graph.name:
        m.graph.name = "graph"
    fix_graph(m.graph, m.graph.name)


def ensure_node_names(m: onnx.ModelProto) -> None:
    for i, n in enumerate(m.graph.node):
        if not n.name:
            n.name = f"{n.op_type}_{i}"


def save_clean(path: str) -> None:
    m = onnx.load(path)
    ensure_graph_names(m)
    ensure_node_names(m)
    onnx.save_model(m, path, save_as_external_data=False)


def preprocess(src: str, dst: str, mode: str) -> str:
    if mode == "none":
        return src
    if mode == "onnx":
        m = onnx.load(src)
        ensure_graph_names(m)
        ensure_node_names(m)
        try:
            m = onnx.shape_inference.infer_shapes(m)
        except Exception:
            pass
        onnx.save_model(m, dst, save_as_external_data=False)
        return dst
    if mode == "ort":
        if _quant_pre_process is None:
            return preprocess(src, dst, "onnx")
        sig = inspect.signature(_quant_pre_process)
        allowed = set(sig.parameters.keys())
        kwargs = {}
        if "skip_symbolic_shape_inference" in allowed:
            kwargs["skip_symbolic_shape_inference"] = True
        if "skip_onnx_shape_inference" in allowed:
            kwargs["skip_onnx_shape_inference"] = False
        if "skip_optimization" in allowed:
            kwargs["skip_optimization"] = False
        try:
            _quant_pre_process(src, dst, **kwargs)
            save_clean(dst)
            return dst
        except Exception:
            return preprocess(src, dst, "onnx")
    raise ValueError(f"Unknown preprocess mode: {mode}")


def pick_calib_method(name: str):
    # fallback to name (str) when CalibrationMethod unavailable
    if CalibrationMethod is None:
        print(f"CalibrationMethod is None, using {name}")
        return name
    return getattr(CalibrationMethod, name, CalibrationMethod.MinMax)


def get_io_names(model_path: str) -> Tuple[List[str], List[str]]:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    outs = [o.name for o in sess.get_outputs()]
    return ins, outs


def onnx_int8_name(src_name: str) -> str:
    return os.path.splitext(src_name)[0] + ".int8.onnx"


def _detect_variable_axis(shapes: List[Tuple[int, ...]]) -> Optional[int]:
    # return axis index if exactly one axis varies across shapes, else None
    if not shapes:
        return None
    nd = len(shapes[0])
    if any(len(s) != nd for s in shapes):
        return None
    var_axes = []
    for ax in range(nd):
        vals = {s[ax] for s in shapes}
        if len(vals) > 1:
            var_axes.append(ax)
    if len(var_axes) == 1:
        return var_axes[0]
    return None


def _crop_center(arr: np.ndarray, axis: int, target: int) -> np.ndarray:
    cur = arr.shape[axis]
    if cur <= target:
        return arr
    start = (cur - target) // 2
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(start, start + target)
    return arr[tuple(sl)]


def _pad(arr: np.ndarray, axis: int, target: int, pad_value: float) -> np.ndarray:
    cur = arr.shape[axis]
    if cur >= target:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target - cur)
    return np.pad(arr, pad_width, mode="constant", constant_values=pad_value)


def _pad_or_crop(arr: np.ndarray, axis: int, target: int, pad_value: float) -> np.ndarray:
    cur = arr.shape[axis]
    if cur > target:
        return _crop_center(arr, axis, target)
    if cur < target:
        return _pad(arr, axis, target, pad_value)
    return arr


def _pad_value_for(name: str, dtype: np.dtype):
    n = name.lower()
    if "mask" in n:
        return 0
    if np.issubdtype(dtype, np.integer):
        return 0
    return 0.0


def _build_pad_plan_percentile(
    folder: str,
    input_names: List[str],
    limit: int,
    pad_percentile: int,
    pad_max: int,
) -> Dict[str, Tuple[int, int]]:
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    files = files[:limit] if limit > 0 else files
    if not files:
        raise RuntimeError(f"No npz in: {folder}")

    shapes_per_in: Dict[str, List[Tuple[int, ...]]] = {n: [] for n in input_names}
    for f in files:
        d = np.load(f, allow_pickle=False)
        for n in input_names:
            if n not in d:
                raise KeyError(f"{f} missing '{n}', keys={list(d.keys())}")
            shapes_per_in[n].append(tuple(d[n].shape))

    plan: Dict[str, Tuple[int, int]] = {}
    for n, shapes in shapes_per_in.items():
        ax = _detect_variable_axis(shapes)
        if ax is None:
            continue
        lens = np.array([s[ax] for s in shapes], dtype=np.int64)
        tgt = int(np.percentile(lens, pad_percentile))
        tgt = max(1, tgt)
        if pad_max > 0:
            tgt = min(tgt, pad_max)
        plan[n] = (ax, tgt)
    return plan


class PaddedNpzDataReader(CalibrationDataReader):
    def __init__(self, folder: str, input_names: List[str], limit: int, pad_percentile: int, pad_max: int):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if limit > 0:
            self.files = self.files[:limit]
        if not self.files:
            raise RuntimeError(f"No calibration npz in: {folder}")
        self.input_names = input_names
        self.pad_plan = _build_pad_plan_percentile(folder, input_names, limit, pad_percentile, pad_max)
        self._iter = iter(self.files)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        try:
            p = next(self._iter)
        except StopIteration:
            return None
        d = np.load(p, allow_pickle=False)
        feeds: Dict[str, np.ndarray] = {}
        for n in self.input_names:
            x = d[n]
            if x.dtype == np.float64:
                x = x.astype(np.float32)
            if n in self.pad_plan:
                axis, tgt = self.pad_plan[n]
                pv = _pad_value_for(n, x.dtype)
                x = _pad_or_crop(x, axis, tgt, pv)
            feeds[n] = x
        return feeds

    def rewind(self) -> None:
        self._iter = iter(self.files)


def safe_copy(src: str, dst: str) -> None:
    shutil.copy2(src, dst)
    try:
        save_clean(dst)
    except Exception:
        pass


def quantize_dynamic_safe(fp32_path: str, out_path: str, op_types: List[str], wt_type: QuantType) -> None:
    try:
        quantize_dynamic(
            model_input=fp32_path,
            model_output=out_path,
            op_types_to_quantize=op_types,
            weight_type=wt_type,
            per_channel=False,
            reduce_range=False,
            use_external_data_format=False,
        )
        save_clean(out_path)
    except Exception as e:
        print(f"[WARN] dynamic quant failed for {os.path.basename(fp32_path)}: {e} -> fallback copy")
        safe_copy(fp32_path, out_path)


def quantize_static_safe(
    fp32_path: str,
    out_path: str,
    calib_folder: str,
    preprocess_mode: str,
    calib_limit: int,
    calibrate_method: str,
    act_type: QuantType,
    wt_type: QuantType,
    per_channel: bool,
    reduce_range: bool,
    op_types: List[str],
    nodes_to_exclude: Optional[List[str]],
    pad_percentile: int,
    pad_max: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="st_q_") as td:
        pre_path = os.path.join(td, "pre.onnx")
        fp32_for_quant = preprocess(fp32_path, pre_path, preprocess_mode)
        ins, _ = get_io_names(fp32_for_quant)

        extra = {"WeightSymmetric": True}
        extra["ActivationSymmetric"] = (act_type == QuantType.QInt8)

        def _run(method: str) -> None:
            sig = inspect.signature(quantize_static)
            allowed = set(sig.parameters.keys())
            kwargs = dict(
                quant_format=QuantFormat.QDQ,
                op_types_to_quantize=op_types,
                per_channel=per_channel,
                reduce_range=reduce_range,
                activation_type=act_type,
                weight_type=wt_type,
                optimize_model=False,
                use_external_data_format=False,
                extra_options=extra,
                calibration_providers=["CPUExecutionProvider"],
            )
            cm = pick_calib_method(method)
            if "calibrate_method" in allowed:
                kwargs["calibrate_method"] = cm
            if nodes_to_exclude and "nodes_to_exclude" in allowed:
                kwargs["nodes_to_exclude"] = nodes_to_exclude
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}

            dr = PaddedNpzDataReader(calib_folder, ins, calib_limit, pad_percentile, pad_max)
            quantize_static(fp32_for_quant, out_path, dr, **kwargs)
            save_clean(out_path)

        try:
            _run(calibrate_method)
        except Exception as e:
            msg = str(e)
            if "inhomogeneous shape" in msg or "setting an array element with a sequence" in msg:
                print(f"[WARN] calib shape issue on {os.path.basename(fp32_path)} -> fallback MinMax")
                _run("MinMax")
            else:
                print(f"[WARN] static quant failed for {os.path.basename(fp32_path)}: {e} -> fallback copy")
                safe_copy(fp32_path, out_path)


def _name_exists(model: onnx.ModelProto, name: str) -> bool:
    for t in model.graph.initializer:
        if t.name == name:
            return True
    for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if v.name == name:
            return True
    for n in model.graph.node:
        if name in n.output:
            return True
    return False


def _unique_name(model: onnx.ModelProto, base: str) -> str:
    if not _name_exists(model, base):
        return base
    i = 0
    while True:
        cand = f"{base}_{i}"
        if not _name_exists(model, cand):
            return cand
        i += 1


def _w8dq_quantize_per_channel_s8(w: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = w.astype(np.float32)
    w_abs = np.max(np.abs(w), axis=tuple(i for i in range(w.ndim) if i != axis), keepdims=False)
    w_abs = np.maximum(w_abs, 1e-8)
    scale = (w_abs / 127.0).astype(np.float32)
    zp = np.zeros_like(scale, dtype=np.int8)

    shape = [1] * w.ndim
    shape[axis] = w.shape[axis]
    scale_b = scale.reshape(shape)
    w_q = np.round(w / scale_b).clip(-127, 127).astype(np.int8)
    return w_q, scale, zp


def apply_w8dq_to_conv_weights(
    model_in: str,
    model_out: str,
    exclude_last_conv: int,
    only_fp32: bool = True,
) -> None:
    m = onnx.load(model_in)
    ensure_graph_names(m)
    ensure_node_names(m)

    convs_all = [n for n in m.graph.node if n.op_type == "Conv"]
    if exclude_last_conv > 0 and len(convs_all) >= exclude_last_conv:
        convs_use = convs_all[:-exclude_last_conv]
    else:
        convs_use = convs_all

    imap = {t.name: t for t in m.graph.initializer}

    def remove_initializer(name: str) -> None:
        keep = [t for t in m.graph.initializer if t.name != name]
        del m.graph.initializer[:]
        m.graph.initializer.extend(keep)

    new_nodes = []
    changed = 0

    for node in m.graph.node:
        if node.op_type != "Conv":
            continue
        if node not in convs_use:
            continue
        if len(node.input) < 2:
            continue

        w_name = node.input[1]
        if w_name not in imap:
            continue
        w_t = imap[w_name]
        w = numpy_helper.to_array(w_t)
        if only_fp32 and w.dtype != np.float32:
            continue

        w_q, scale, zp = _w8dq_quantize_per_channel_s8(w, axis=0)

        wq_name = _unique_name(m, w_name + "_wq")
        sc_name = _unique_name(m, w_name + "_scale")
        zp_name = _unique_name(m, w_name + "_zp")
        dq_out = _unique_name(m, w_name + "_dq")

        m.graph.initializer.extend([numpy_helper.from_array(w_q, name=wq_name)])
        m.graph.initializer.extend([numpy_helper.from_array(scale.astype(np.float32), name=sc_name)])
        m.graph.initializer.extend([numpy_helper.from_array(zp.astype(np.int8), name=zp_name)])

        dq = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=[wq_name, sc_name, zp_name],
            outputs=[dq_out],
            name=_unique_name(m, "DQ_" + w_name),
            axis=0,
        )
        new_nodes.append(dq)

        node.input[1] = dq_out
        remove_initializer(w_name)
        changed += 1

    if new_nodes:
        old_nodes = list(m.graph.node)
        del m.graph.node[:]
        m.graph.node.extend(new_nodes + old_nodes)

    onnx.checker.check_model(m)
    onnx.save_model(m, model_out, save_as_external_data=False)
    save_clean(model_out)
    print(f"[W8-DQ] conv weights compressed: {changed} (exclude_last_conv={exclude_last_conv})")


def infer_vocoder_latent_shape(vocoder_fp32: str, voc_calib_dir: str) -> Optional[Tuple[int, ...]]:
    try:
        voc_in, _ = get_io_names(vocoder_fp32)
        if len(voc_in) != 1:
            return None
        inp = voc_in[0]
        files = sorted(glob.glob(os.path.join(voc_calib_dir, "*.npz")))
        if not files:
            return None
        d = np.load(files[0], allow_pickle=False)
        if inp not in d:
            return None
        return tuple(d[inp].shape)
    except Exception:
        return None


def pick_ve_output_index(ve_model_path: str, ve_calib_dir: str, voc_latent_shape: Optional[Tuple[int, ...]]) -> int:
    ve_in, _ = get_io_names(ve_model_path)
    files = sorted(glob.glob(os.path.join(ve_calib_dir, "*.npz")))
    if not files:
        return 0
    d = np.load(files[0], allow_pickle=False)
    feeds = {}
    for n in ve_in:
        x = d[n]
        if x.dtype == np.float64:
            x = x.astype(np.float32)
        feeds[n] = x

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(ve_model_path, sess_options=so, providers=["CPUExecutionProvider"])
    outs = sess.run(None, feeds)

    best = 0
    if voc_latent_shape is not None:
        vrank = len(voc_latent_shape)
        for i, y in enumerate(outs):
            y = np.asarray(y)
            if not np.issubdtype(y.dtype, np.floating):
                continue
            if y.ndim != vrank:
                continue
            # Supertonic VE latent dim 512, pick output matching vocoder input
            if 512 in y.shape:
                best = i
                break
        return best

    for i, y in enumerate(outs):
        y = np.asarray(y)
        if np.issubdtype(y.dtype, np.floating) and y.ndim == 3 and (512 in y.shape):  # latent dim
            best = i
            break
    return best


def build_vocoder_calib_from_ve(
    ve_model_path: str,
    ve_calib_dir: str,
    vocoder_fp32: str,
    out_dir: str,
    ve_output_index: int,
    limit: int,
    pad_percentile: int,
    pad_max: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    voc_in, _ = get_io_names(vocoder_fp32)
    if len(voc_in) != 1:
        raise RuntimeError(f"vocoder inputs != 1, got {voc_in}")
    voc_in_name = voc_in[0]

    ve_in, _ = get_io_names(ve_model_path)
    files = sorted(glob.glob(os.path.join(ve_calib_dir, "*.npz")))
    files = files[:limit] if limit > 0 else files
    if not files:
        raise RuntimeError(f"No npz in {ve_calib_dir}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(ve_model_path, sess_options=so, providers=["CPUExecutionProvider"])

    ve_pad_plan = _build_pad_plan_percentile(ve_calib_dir, ve_in, limit, pad_percentile, pad_max)

    latents = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        feeds = {}
        for n in ve_in:
            x = d[n]
            if x.dtype == np.float64:
                x = x.astype(np.float32)
            if n in ve_pad_plan:
                axis, tgt = ve_pad_plan[n]
                pv = _pad_value_for(n, x.dtype)
                x = _pad_or_crop(x, axis, tgt, pv)
            feeds[n] = x
        y = np.asarray(sess.run(None, feeds)[ve_output_index], dtype=np.float32)
        latents.append(y)

    shapes = [tuple(z.shape) for z in latents]
    ax = _detect_variable_axis(shapes)
    if ax is not None:
        lens = np.array([s[ax] for s in shapes], dtype=np.int64)
        tgt = int(np.percentile(lens, pad_percentile))
        tgt = max(1, tgt)
        if pad_max > 0:
            tgt = min(tgt, pad_max)
        latents = [_pad_or_crop(z, ax, tgt, 0.0) for z in latents]

    for i, y in enumerate(latents):
        np.savez(os.path.join(out_dir, f"{i:05d}.npz"), **{voc_in_name: y})


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-dir", type=str, required=True, help="source model dir"
    )
    parser.add_argument(
        "--dst-dir", type=str, required=True, help="output model dir"
    )
    parser.add_argument(
        "--calib-dir", type=str, required=True, help="calibration npz dir"
    )
    parser.add_argument(
        "--preprocess", choices=["onnx", "ort", "none"], default="ort"
    )
    parser.add_argument("--duration-predictor", default="duration_predictor.onnx")
    parser.add_argument("--text-encoder", default="text_encoder.onnx")
    parser.add_argument("--vector-estimator", default="vector_estimator.onnx")
    parser.add_argument("--vocoder", default="vocoder.onnx")
    parser.add_argument("--dp-mode", choices=["copy", "dynamic"], default="copy")
    parser.add_argument("--te-mode", choices=["copy", "dynamic"], default="copy")
    parser.add_argument(
        "--dp-te-weight-type", choices=["qint8", "quint8"], default="qint8"
    )
    parser.add_argument("--ve-mode", choices=["copy", "dynamic"], default="dynamic")
    parser.add_argument("--ve-conv-w8dq", action="store_true", default=True)
    parser.add_argument("--ve-w8dq-exclude-last-conv", type=int, default=6)
    parser.add_argument("--ve-calib-limit", type=int, default=100)
    parser.add_argument("--vocoder-calib-limit", type=int, default=100)
    parser.add_argument(
        "--vocoder-calibrate-method",
        choices=["MinMax", "Entropy", "Percentile"],
        default="Percentile",
    )
    parser.add_argument("--vocoder-act", choices=["qint8", "quint8"], default="quint8")
    parser.add_argument("--vocoder-wt", choices=["qint8", "quint8"], default="qint8")
    parser.add_argument("--vocoder-per-channel", action="store_true", default=True)
    parser.add_argument("--vocoder-reduce-range", action="store_true", default=True)
    parser.add_argument("--exclude-last-conv", type=int, default=8)
    parser.add_argument("--vocoder-tail-w8dq", action="store_true", default=True)
    parser.add_argument(
        "--vocoder-tail-w8dq-exclude-last-conv", type=int, default=0
    )
    parser.add_argument("--vocoder-calib-from-ve", action="store_true", default=True)
    parser.add_argument("--ve-output-index", type=int, default=-1)
    parser.add_argument("--pad-percentile", type=int, default=90)
    parser.add_argument("--pad-max", type=int, default=0)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.dst_dir, exist_ok=True)

    print("ORT:", ort.__version__, "providers:", ort.get_available_providers())

    dp_fp32 = os.path.join(args.src_dir, args.duration_predictor)
    te_fp32 = os.path.join(args.src_dir, args.text_encoder)
    ve_fp32 = os.path.join(args.src_dir, args.vector_estimator)
    voc_fp32 = os.path.join(args.src_dir, args.vocoder)

    dp_out = os.path.join(args.dst_dir, onnx_int8_name(args.duration_predictor))
    te_out = os.path.join(args.dst_dir, onnx_int8_name(args.text_encoder))
    ve_out = os.path.join(args.dst_dir, onnx_int8_name(args.vector_estimator))
    voc_out = os.path.join(args.dst_dir, onnx_int8_name(args.vocoder))

    dp_te_wt = QuantType.QInt8 if args.dp_te_weight_type == "qint8" else QuantType.QUInt8
    voc_act = QuantType.QInt8 if args.vocoder_act == "qint8" else QuantType.QUInt8
    voc_wt = QuantType.QInt8 if args.vocoder_wt == "qint8" else QuantType.QUInt8

    if args.dp_mode == "copy":
        safe_copy(dp_fp32, dp_out)
    else:
        quantize_dynamic_safe(dp_fp32, dp_out, ["MatMul", "Gemm"], dp_te_wt)

    if args.te_mode == "copy":
        safe_copy(te_fp32, te_out)
    else:
        quantize_dynamic_safe(te_fp32, te_out, ["MatMul", "Gemm"], dp_te_wt)

    if args.ve_mode == "copy":
        safe_copy(ve_fp32, ve_out)
    else:
        quantize_dynamic_safe(ve_fp32, ve_out, ["MatMul", "Gemm"], QuantType.QInt8)

    if args.ve_conv_w8dq:
        apply_w8dq_to_conv_weights(
            model_in=ve_out,
            model_out=ve_out,
            exclude_last_conv=args.ve_w8dq_exclude_last_conv,
            only_fp32=True,
        )

    ve_calib = os.path.join(args.calib_dir, os.path.splitext(args.vector_estimator)[0])
    voc_calib_dir = os.path.join(args.calib_dir, os.path.splitext(args.vocoder)[0])

    voc_lat_shape = infer_vocoder_latent_shape(voc_fp32, voc_calib_dir)

    nodes_excl = None
    if args.exclude_last_conv > 0:
        with tempfile.TemporaryDirectory(prefix="voc_pre_") as td:
            pre_voc = os.path.join(td, "voc_pre.onnx")
            voc_for = preprocess(voc_fp32, pre_voc, args.preprocess)
            m = onnx.load(voc_for)
            ensure_node_names(m)
            convs = [n.name for n in m.graph.node if n.op_type == "Conv"]
            if len(convs) >= args.exclude_last_conv:
                nodes_excl = convs[-args.exclude_last_conv:]

    def _run_vocoder_quantize(calib_folder: str) -> None:
        quantize_static_safe(
            fp32_path=voc_fp32,
            out_path=voc_out,
            calib_folder=calib_folder,
            preprocess_mode=args.preprocess,
            calib_limit=args.vocoder_calib_limit,
            calibrate_method=args.vocoder_calibrate_method,
            act_type=voc_act,
            wt_type=voc_wt,
            per_channel=args.vocoder_per_channel,
            reduce_range=args.vocoder_reduce_range,
            op_types=["Conv"],
            nodes_to_exclude=nodes_excl,
            pad_percentile=args.pad_percentile,
            pad_max=args.pad_max,
        )
        if args.vocoder_tail_w8dq and args.exclude_last_conv > 0:
            apply_w8dq_to_conv_weights(
                model_in=voc_out,
                model_out=voc_out,
                exclude_last_conv=args.vocoder_tail_w8dq_exclude_last_conv,
                only_fp32=True,
            )

    if args.vocoder_calib_from_ve:
        with tempfile.TemporaryDirectory(prefix="vocoder_calib_") as tmp_voc_calib:
            ve_idx = args.ve_output_index
            if ve_idx < 0:
                ve_idx = pick_ve_output_index(ve_out, ve_calib, voc_lat_shape)
            print(f"[INFO] VE output index for vocoder calib: {ve_idx}")
            build_vocoder_calib_from_ve(
                ve_model_path=ve_out,
                ve_calib_dir=ve_calib,
                vocoder_fp32=voc_fp32,
                out_dir=tmp_voc_calib,
                ve_output_index=ve_idx,
                limit=args.vocoder_calib_limit,
                pad_percentile=args.pad_percentile,
                pad_max=args.pad_max,
            )
            _run_vocoder_quantize(tmp_voc_calib)
    else:
        _run_vocoder_quantize(voc_calib_dir)

    print("Quantization completed!")


if __name__ == "__main__":
    main()
