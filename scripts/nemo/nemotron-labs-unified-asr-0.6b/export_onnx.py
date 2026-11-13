#!/usr/bin/env python3
# Copyright (c) 2026 Julian Pscheid
"""Export nvidia/nemotron-labs-unified-asr-0.6b for buffered streaming."""

import argparse
import json
from pathlib import Path

import onnx
import torch
from huggingface_hub import hf_hub_download
from onnx import compose, helper
from onnxruntime.quantization import QuantType, quantize_dynamic


class LanguagePrompt(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.prompt_kernel = model.prompt_kernel
        self.num_prompts = model.num_prompts

    def forward(self, encoded, prompt_index):
        x = encoded.transpose(1, 2)
        prompt = torch.nn.functional.one_hot(prompt_index, self.num_prompts)
        prompt = prompt.to(x.dtype).unsqueeze(1).expand(-1, x.shape[1], -1)
        return self.prompt_kernel(torch.cat([x, prompt], dim=-1)).transpose(1, 2)


def add_meta_data(model, metadata):
    del model.metadata_props[:]
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = str(value)


def merge_prompt(encoder_path, prompt_path, output_path, metadata):
    # Leave the fp32 encoder's external weights on disk when composing. Loading
    # them would exceed protobuf's 2 GB limit during merge_models validation.
    encoder = onnx.load(encoder_path, load_external_data=False)
    prompt = onnx.load(prompt_path)
    opsets = {}
    for model in (encoder, prompt):
        for op in model.opset_import:
            opsets[op.domain] = max(opsets.get(op.domain, 0), op.version)
    for model in (encoder, prompt):
        del model.opset_import[:]
        model.opset_import.extend(helper.make_opsetid(k, v) for k, v in opsets.items())
    prompt.ir_version = encoder.ir_version
    prompt = compose.add_prefix(prompt, "prompt_")
    encoded, length = [out.name for out in encoder.graph.output][:2]
    # Keep prompt_index unprefixed to match the cache-aware NeMo encoder API.
    for node in prompt.graph.node:
        for i, name in enumerate(node.input):
            if name == "prompt_prompt_index":
                node.input[i] = "prompt_index"
    prompt.graph.input[1].name = "prompt_index"
    merged = compose.merge_models(
        encoder,
        prompt,
        io_map=[(encoded, "prompt_encoded")],
        outputs=["prompt_output", length],
    )
    # merge_models filters graph outputs in their original order; its outputs
    # argument does not reorder them. The runtime expects acoustic output first.
    outputs = {value.name: value for value in merged.graph.output}
    encoded_output, length_output = outputs["prompt_output"], outputs[length]
    if (
        encoded_output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or len(encoded_output.type.tensor_type.shape.dim) != 3
        or length_output.type.tensor_type.elem_type != onnx.TensorProto.INT64
        or len(length_output.type.tensor_type.shape.dim) != 1
    ):
        raise ValueError("Expected encoder outputs float [B,D,T] and int64 [B]")
    del merged.graph.output[:]
    merged.graph.output.extend([encoded_output, length_output])
    add_meta_data(merged, metadata)
    onnx.save(merged, output_path)
    onnx.checker.check_model(str(output_path))


@torch.no_grad()
def main():
    import nemo.collections.asr as nemo_asr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        help="Local .nemo checkpoint; otherwise download the pinned public model.",
    )
    parser.add_argument(
        "--left-context",
        type=int,
        default=70,
        help="Left context in 80 ms encoder frames (70 = 5.6 s).",
    )
    args = parser.parse_args()
    if args.left_context < 0:
        parser.error("--left-context must be nonnegative")
    checkpoint = args.model or hf_hub_download(
        "nvidia/nemotron-labs-unified-asr-0.6b",
        "nemotron-labs-unified-asr-0.6b.nemo",
        revision="023ce907267a09a308a5cb69239447cb3d0c1597",
    )
    model = nemo_asr.models.ASRModel.restore_from(checkpoint, map_location="cpu")
    model.eval()
    model.cfg.validation_ds = dict()
    model.encoder.set_default_att_context_size([args.left_context, 7, 7])
    if not hasattr(model, "prompt_kernel") or not hasattr(model, "num_prompts"):
        raise ValueError("Expected a multilingual model with prompt_kernel")
    prompts = {
        str(k): int(v) for k, v in model.cfg.model_defaults.prompt_dictionary.items()
    }
    if "auto" not in prompts or any(
        v < 0 or v >= model.num_prompts for v in prompts.values()
    ):
        raise ValueError("Invalid language prompt dictionary")
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i, token in enumerate(model.joint.vocabulary):
            f.write(f"{token} {i}\n")
        f.write(f"<blk> {i + 1}\n")
    model.encoder.export("encoder.raw.onnx")
    raw_encoder = onnx.load("encoder.raw.onnx")
    Path("encoder.weights").unlink(missing_ok=True)
    onnx.save(
        raw_encoder,
        "encoder.raw.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="encoder.weights",
    )
    del raw_encoder
    model.decoder.export("decoder.onnx")
    model.joint.export("joiner.onnx")
    # Unsigned weights avoid x86 u8s8 saturation without VNNI; ARM is unaffected.
    # Quantize acoustic MatMul ops before attaching the fp32 language prompt.
    quantize_dynamic(
        model_input="encoder.raw.onnx",
        model_output="encoder.raw.int8.onnx",
        weight_type=QuantType.QUInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
    )
    encoder_dim = model.encoder._feat_out
    torch.onnx.export(
        LanguagePrompt(model),
        (torch.randn(1, encoder_dim, 7), torch.tensor([prompts["auto"]])),
        "prompt.onnx",
        input_names=["encoded", "prompt_index"],
        output_names=["output"],
        dynamic_axes={
            "encoded": {0: "B", 2: "T"},
            "prompt_index": {0: "B"},
            "output": {0: "B", 2: "T"},
        },
        opset_version=17,
        dynamo=False,
    )
    factor = model.encoder.subsampling_factor
    normalize = model.cfg.preprocessor.normalize
    metadata = {
        "vocab_size": model.decoder.vocab_size,
        "normalize_type": "" if normalize == "NA" else normalize,
        "pred_rnn_layers": model.decoder.pred_rnn_layers,
        "pred_hidden": model.decoder.pred_hidden,
        "subsampling_factor": factor,
        "model_type": "EncDecRNNTBPEModel",
        "streaming_model_type": "nemo_parakeet_unified_streaming",
        "buffered_streaming": 1,
        "left_encoder_frames": args.left_context,
        "chunk_encoder_frames": 7,
        "right_encoder_frames": 7,
        "left_feature_frames": args.left_context * factor,
        "chunk_feature_frames": 7 * factor,
        "right_feature_frames": 7 * factor,
        "version": "2",
        "model_author": "NeMo",
        "url": "https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b",
        "feat_dim": model.cfg.preprocessor.features,
        "prompt_dictionary": json.dumps(prompts, sort_keys=True),
        "language_prompt_ids": json.dumps(prompts, sort_keys=True),
        "auto_prompt_id": prompts["auto"],
    }
    merge_prompt("encoder.raw.onnx", "prompt.onnx", "encoder.onnx", metadata)
    merge_prompt("encoder.raw.int8.onnx", "prompt.onnx", "encoder.int8.onnx", metadata)
    decoder = onnx.load("decoder.onnx")
    add_meta_data(decoder, {"streaming_model_type": "nemo_parakeet_unified_streaming"})
    onnx.save(decoder, "decoder.onnx")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
