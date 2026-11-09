#!/usr/bin/env python3
# Copyright      2026  danielxxomg
#
# Exports Moondream's Parakeet Ultra to sherpa-onnx. Ultra is a post-trained
# nvidia/parakeet-tdt-0.6b-v3: same architecture, same tokenizer, same tensor
# shapes, different weights. So v3's NeMo model supplies the architecture and
# Ultra's safetensors supply the weights.
#
# The tensor rename table was independently reimplemented from the tensor
# names used by the public conversion at
# https://huggingface.co/leogaudin/sherpa-onnx-nemo-parakeet-ultra-int8
# (CC-BY-4.0); the export recipe follows sherpa-onnx's Apache-2.0 scripts.

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict

import nemo.collections.asr as nemo_asr
import onnx
import torch
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from safetensors.torch import load_file

# Add parent directory to path to import generate_bpe_vocab
sys.path.insert(0, str(Path(__file__).parent.parent))
from generate_bpe_vocab import generate_bpe_vocab_from_model

ULTRA = "moondream/parakeet-ultra"
V3 = "nvidia/parakeet-tdt-0.6b-v3"
# sherpa-onnx takes a model for a TDT one (its outputs include durations) only
# if this metadata contains "tdt"; otherwise it loads it as a plain transducer.
URL = f"https://huggingface.co/{ULTRA} (tdt, trained from {V3})"

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


def ultra_state_for(model, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Ultra's tensors under NeMo's names, in v3's dtype; stops on anything that does not line up."""
    expected = {k: v for k, v in model.state_dict().items() if not KEPT.fullmatch(k)}
    mapped: Dict[str, torch.Tensor] = {}
    unexpected, ignored = [], []
    for name, tensor in weights.items():
        if IGNORED.fullmatch(name):
            ignored.append(name)
            continue
        target = rename(name)
        if target is None or target not in expected:
            unexpected.append(name)
            continue
        if target in mapped:
            raise SystemExit(f"two weights map to {target}")
        if tuple(tensor.shape) != tuple(expected[target].shape):
            raise SystemExit(
                f"{name} -> {target}: shape {tuple(tensor.shape)}, "
                f"v3 has {tuple(expected[target].shape)}"
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
    differing = [i for i in range(len(v3)) if ids.get(i) != v3[i]]
    if differing or len(ids) != len(v3) + 1:
        raise SystemExit(
            f"vocabulary differs from v3's at {differing[:10]} "
            f"({len(ids)} against {len(v3) + 1})"
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


def float32_convs(filename: str) -> list:
    """Encoder convolutions that stay float32 in the int8 model.

    quantize_dynamic uses one uint8 scale per weight tensor. The 24 depthwise
    convolutions carry a folded BatchNorm, whose per-channel scales are far
    apart, so a single scale per tensor loses most of their precision; the 5
    subsampling (pre_encode) convolutions are kept float32 too. All 29 are
    small, so the int8 model keeps its size and speed.
    """
    model = onnx.load(filename, load_external_data=False)
    names = [
        n.name
        for n in model.graph.node
        if n.op_type == "Conv"
        and ("/depthwise_conv/" in n.name or n.name.startswith("/pre_encode/"))
    ]
    # quantize_dynamic silently ignores names it does not find, so a renamed
    # node would be quantized again without any error
    assert len(names) == 29, names
    return names


def local_or_download(local: str, repo: str, name: str) -> Path:
    path = Path(local)
    return path if path.is_file() else Path(hf_hub_download(repo, name))


@torch.no_grad()
def main():
    nemo_path = local_or_download(
        "./parakeet-tdt-0.6b-v3.nemo", V3, "parakeet-tdt-0.6b-v3.nemo"
    )
    weights_path = local_or_download("./model.safetensors", ULTRA, "model.safetensors")
    tokenizer_path = local_or_download("./tokenizer.json", ULTRA, "tokenizer.json")

    asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=str(nemo_path))
    check_vocabulary(asr_model, tokenizer_path)
    state = ultra_state_for(asr_model, load_file(str(weights_path)))
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
            nodes_to_exclude=float32_convs(f"./{m}.onnx") if m == "encoder" else [],
        )
        os.system("ls -lh *.onnx")

    add_meta_data("encoder.int8.onnx", meta_data)
    add_meta_data("encoder.onnx", meta_data)
    print("meta_data", meta_data)


if __name__ == "__main__":
    main()
