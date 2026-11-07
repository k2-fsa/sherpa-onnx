#!/usr/bin/env python3
# Copyright      2026  danielxxomg
#
# Exports Moondream's Parakeet Redux to sherpa-onnx. Redux is a ternary
# (1.58-bit) version of nvidia/parakeet-tdt-0.6b-v3: same architecture, same
# tokenizer, same 8193-token vocabulary, different weights. So v3's NeMo model
# supplies the architecture and Redux's safetensors supply the weights.
#
# Every encoder linear and pointwise conv weight is packed in the
# thrush-ternary-v2 format specified by ternary.json in moondream/parakeet-redux:
# base 3, 5 elements per byte, row-major, least significant digit first, pad
# digits 0, and w = scales[row, col // 128] * (code - 1) with code in {0, 1, 2}.
# The remaining tensors are dense and carry the same names as
# moondream/parakeet-ultra's, so the tensor rename table below is shared with
# scripts/nemo/parakeet-tdt-0.6b-ultra/export_onnx.py unchanged.
#
# The rename table was independently reimplemented from the tensor names used by
# the public conversion at
# https://huggingface.co/leogaudin/sherpa-onnx-nemo-parakeet-ultra-int8
# (CC-BY-4.0); the export recipe follows sherpa-onnx's Apache-2.0 scripts.

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import nemo.collections.asr as nemo_asr
import onnx
import torch
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from safetensors.torch import load_file

# Add parent directory to path to import generate_bpe_vocab
sys.path.insert(0, str(Path(__file__).parent.parent))
from generate_bpe_vocab import generate_bpe_vocab_from_model

REDUX = "moondream/parakeet-redux"
V3 = "nvidia/parakeet-tdt-0.6b-v3"
# Pinned so a re-run publishes the weights this script was tested with.
REDUX_REVISION = "2bf128600aac4b16946f7ed8372e56117fe5e23b"
V3_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
# sherpa-onnx takes a model for a TDT one (its outputs include durations) only
# if this metadata contains "tdt"; otherwise it loads it as a plain transducer.
URL = f"https://huggingface.co/{REDUX} (ternary, trained from {V3})"

# The packing documented by ternary.json. Anything else is a different format
# and must not be decoded with this arithmetic.
TERNARY_FORMAT = "thrush-ternary-v2"
TERNARY_BASE = 3
ELEMENTS_PER_BYTE = 5

# transformers name -> NeMo name, as regular expressions on the whole name.
RENAMES = [
    (r"encoder\.subsampling\.layers\.(\d+)\.(weight|bias)", r"encoder.pre_encode.conv.\1.\2"),
    (r"encoder\.subsampling\.linear\.(weight|bias)", r"encoder.pre_encode.out.\1"),
    (r"encoder\.layers\.(\d+)\.conv\.norm\.(.+)", r"encoder.layers.\1.conv.batch_norm.\2"),
    (r"encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight", r"encoder.layers.\1.self_attn.linear_q.weight"),
    (r"encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight", r"encoder.layers.\1.self_attn.linear_k.weight"),
    (r"encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight", r"encoder.layers.\1.self_attn.linear_v.weight"),
    (r"encoder\.layers\.(\d+)\.self_attn\.o_proj\.weight", r"encoder.layers.\1.self_attn.linear_out.weight"),
    (r"encoder\.layers\.(\d+)\.self_attn\.relative_k_proj\.weight", r"encoder.layers.\1.self_attn.linear_pos.weight"),
    (r"encoder\.layers\.(\d+)\.self_attn\.bias_u", r"encoder.layers.\1.self_attn.pos_bias_u"),
    (r"encoder\.layers\.(\d+)\.self_attn\.bias_v", r"encoder.layers.\1.self_attn.pos_bias_v"),
    (r"encoder_projector\.(weight|bias)", r"joint.enc.\1"),
    (r"decoder\.decoder_projector\.(weight|bias)", r"joint.pred.\1"),
    (r"decoder\.embedding\.weight", r"decoder.prediction.embed.weight"),
    (r"decoder\.lstm\.(.+)", r"decoder.prediction.dec_rnn.lstm.\1"),
    (r"joint\.head\.(weight|bias)", r"joint.joint_net.2.\1"),
]
# Moondream's voice-activity head: not part of the transcriber, and the only
# tensors NeMo has no place for.
IGNORED = re.compile(r"vad_head\..+")
# Kept as v3 has them: the feature extractor's window and filterbank, which
# have no weights to train.
KEPT = re.compile(r"preprocessor\..+")


def rename(name: str):
    for pattern, replacement in RENAMES:
        if re.fullmatch(pattern, name):
            return re.sub(pattern, replacement, name)
    return name if re.fullmatch(r"encoder\.layers\.\d+\.[a-z_0-9.]+", name) else None


def decode_ternary(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    in_features: int,
    group_size: int,
) -> torch.Tensor:
    """Decode one thrush-ternary-v2 packed module into its dense weight matrix."""
    if qweight.dtype != torch.uint8:
        raise SystemExit(f"qweight is {qweight.dtype}, expected uint8")
    out_features, row_bytes = qweight.shape
    expected_row_bytes = -(-in_features // ELEMENTS_PER_BYTE)
    if row_bytes != expected_row_bytes:
        raise SystemExit(
            f"row_bytes {row_bytes} != ceil({in_features}/{ELEMENTS_PER_BYTE})"
        )
    expected_scales = (out_features, -(-in_features // group_size))
    if tuple(scales.shape) != expected_scales:
        raise SystemExit(
            f"scales {tuple(scales.shape)} != {expected_scales} "
            f"for {out_features}x{in_features} in groups of {group_size}"
        )

    packed = qweight.to(torch.int32)
    # Element i of a row is base-3 digit i % 5 of byte i // 5, least significant
    # digit first; the padding of the last byte decodes to zero weight, so it is
    # dropped by the slice below.
    digits = [
        (packed // (TERNARY_BASE**i)) % TERNARY_BASE
        for i in range(ELEMENTS_PER_BYTE)
    ]
    codes = torch.stack(digits, dim=2).reshape(
        out_features, row_bytes * ELEMENTS_PER_BYTE
    )[:, :in_features]

    codes = codes.to(torch.float32) - 1.0
    repeated = scales.to(torch.float32).repeat_interleave(group_size, dim=1)[
        :, :in_features
    ]
    return repeated * codes


def read_ternary_spec(path: Path) -> Dict:
    spec = json.loads(path.read_text())
    if spec.get("format") != TERNARY_FORMAT:
        raise SystemExit(
            f"{path}: format {spec.get('format')!r}, expected {TERNARY_FORMAT!r}"
        )
    return spec


def unpack_weights(
    weights: Dict[str, torch.Tensor], spec: Dict
) -> Tuple[Dict[str, torch.Tensor], Set[str]]:
    """Redux's safetensors with every packed module decoded to a dense `<name>.weight`.

    Also returns the decoded names that are conv1d pointwise weights, which NeMo
    keeps with a trailing unit dimension the 2D packing does not carry.
    """
    modules = spec["quantized_modules"]
    default_group = spec["quant"]["group_size"]
    packed = {m["name"] + suffix for m in modules for suffix in (".qweight", ".scales")}
    missing = sorted(packed - set(weights))
    if missing:
        raise SystemExit(f"ternary.json names tensors that are not in the file: {missing[:10]}")

    dense: Dict[str, torch.Tensor] = {}
    conv1d: Set[str] = set()
    for module in modules:
        name = module["name"] + ".weight"
        dense[name] = decode_ternary(
            weights[module["name"] + ".qweight"],
            weights[module["name"] + ".scales"],
            module["in_features"],
            module.get("group_size", default_group),
        )
        if module["as_conv1d"]:
            conv1d.add(name)

    stray = sorted(
        k for k in weights if k.endswith((".qweight", ".scales")) and k not in packed
    )
    if stray:
        raise SystemExit(f"packed tensors that ternary.json does not list: {stray[:10]}")

    for name, tensor in weights.items():
        if name in packed:
            continue
        if name in dense:
            raise SystemExit(f"{name} is both packed and dense in the file")
        dense[name] = tensor

    print(f"{len(modules)} ternary modules decoded ({len(conv1d)} of them conv1d)")
    return dense, conv1d


def redux_state_for(
    model, weights: Dict[str, torch.Tensor], spec: Dict
) -> Dict[str, torch.Tensor]:
    """Redux's tensors under NeMo's names, in v3's dtype; stops on anything that does not line up."""
    expected = {k: v for k, v in model.state_dict().items() if not KEPT.fullmatch(k)}
    dense, conv1d = unpack_weights(weights, spec)
    mapped: Dict[str, torch.Tensor] = {}
    unexpected, ignored = [], []
    for name, tensor in dense.items():
        if IGNORED.fullmatch(name):
            ignored.append(name)
            continue
        target = rename(name)
        if target is None or target not in expected:
            unexpected.append(name)
            continue
        if target in mapped:
            raise SystemExit(f"two weights map to {target}")
        target_shape = tuple(expected[target].shape)
        if name in conv1d:
            if tensor.dim() != 2:
                raise SystemExit(f"{name}: packed conv1d decoded to {tensor.dim()}D")
            if (
                len(target_shape) != 3
                or target_shape[-1] != 1
                or target_shape[:2] != tuple(tensor.shape)
            ):
                raise SystemExit(
                    f"{name} -> {target}: 2D {tuple(tensor.shape)} is not the "
                    f"trailing-unit 3D conv weight v3 has {target_shape}"
                )
            tensor = tensor.unsqueeze(-1)
        elif tuple(tensor.shape) != target_shape:
            raise SystemExit(
                f"{name} -> {target}: shape {tuple(tensor.shape)}, "
                f"v3 has {target_shape}"
            )
        mapped[target] = tensor.to(expected[target].dtype)
    missing = sorted(set(expected) - set(mapped))
    if unexpected or missing:
        raise SystemExit(
            f"unexpected: {unexpected[:10]} ({len(unexpected)}); "
            f"missing: {missing[:10]} ({len(missing)})"
        )
    print(f"{len(mapped)} weights renamed; ignored {len(ignored)}: {sorted(set(i.split('.')[0] for i in ignored))}")
    return mapped


def check_vocabulary(model, tokenizer_json: Path) -> None:
    tokenizer = json.loads(tokenizer_json.read_text())
    ids = {i: t for t, i in tokenizer["model"]["vocab"].items()}
    ids.update({a["id"]: a["content"] for a in tokenizer["added_tokens"]})
    v3 = dict(enumerate(model.joint.vocabulary))
    # The blank is the last id, named differently by each (`<blank>`, `<blk>`).
    # tokens.txt writes it at len(v3), so it has to be there and nowhere else.
    differing = [i for i in range(len(v3)) if ids.get(i) != v3[i]]
    if differing or len(ids) != len(v3) + 1 or len(v3) not in ids:
        raise SystemExit(
            f"vocabulary differs from v3's at {differing[:10]} "
            f"({len(ids)} against {len(v3) + 1}); "
            f"blank id {len(v3)} present: {len(v3) in ids}"
        )
    print(f"vocabulary: the same {len(v3)} tokens as v3, in the same order")


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    if filename == "encoder.onnx":
        onnx.save(
            model,
            filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="encoder.weights",
        )
    else:
        onnx.save(model, filename)


def local_or_download(local: str, repo: str, name: str, revision: str) -> Path:
    path = Path(local)
    return path if path.is_file() else Path(hf_hub_download(repo, name, revision=revision))


@torch.no_grad()
def main():
    nemo_path = local_or_download(
        "./parakeet-tdt-0.6b-v3.nemo", V3, "parakeet-tdt-0.6b-v3.nemo", V3_REVISION
    )
    weights_path = local_or_download(
        "./model.safetensors", REDUX, "model.safetensors", REDUX_REVISION
    )
    tokenizer_path = local_or_download(
        "./tokenizer.json", REDUX, "tokenizer.json", REDUX_REVISION
    )
    ternary_path = local_or_download(
        "./ternary.json", REDUX, "ternary.json", REDUX_REVISION
    )

    asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=str(nemo_path))
    check_vocabulary(asr_model, tokenizer_path)
    state = redux_state_for(
        asr_model, load_file(str(weights_path)), read_ternary_spec(ternary_path)
    )
    result = asr_model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(
        not KEPT.fullmatch(k) for k in result.missing_keys
    ):
        raise SystemExit(f"load_state_dict: {result}")
    asr_model.eval()

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i + 1}\n")
        print("Saved to tokens.txt")

    # Generate bpe.vocab for hotword support
    print("Generating bpe.vocab for hotword support...")
    generate_bpe_vocab_from_model(
        asr_model=asr_model,
        output_path="./bpe.vocab",
    )

    asr_model.encoder.export("encoder.onnx")
    asr_model.decoder.export("decoder.onnx")
    asr_model.joint.export("joiner.onnx")
    os.system("ls -lh *.onnx")

    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "normalize_type": normalize_type,
        "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
        "pred_hidden": asr_model.decoder.pred_hidden,
        "subsampling_factor": 8,
        "model_type": "EncDecRNNTBPEModel",
        "version": "2",
        "model_author": "Moondream",
        "url": URL,
        "comment": "Only the transducer branch is exported",
        "feat_dim": 128,
    }

    for m in ["encoder", "decoder", "joiner"]:
        quantize_dynamic(
            model_input=f"./{m}.onnx",
            model_output=f"./{m}.int8.onnx",
            weight_type=QuantType.QUInt8 if m == "encoder" else QuantType.QInt8,
        )
        os.system("ls -lh *.onnx")

    add_meta_data("encoder.int8.onnx", meta_data)
    add_meta_data("encoder.onnx", meta_data)
    print("meta_data", meta_data)


if __name__ == "__main__":
    main()
