#!/usr/bin/env python3
# Copyright      2026  Julian Pscheid
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import nemo.collections.asr as nemo_asr
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

ENCODER_INPUT_NAMES = [
    "audio_signal",
    "length",
    "cache_last_channel",
    "cache_last_time",
    "cache_last_channel_len",
    "prompt_index",
]

ENCODER_OUTPUT_NAMES = [
    "outputs",
    "encoded_lengths",
    "cache_last_channel_next",
    "cache_last_time_next",
    "cache_last_channel_next_len",
]

FORWARD_FOR_EXPORT_ARGS = [
    "audio_signal",
    "length",
    "cache_last_channel",
    "cache_last_time",
    "cache_last_channel_len",
]


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place."""
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    external_filename = filename.split(".onnx")[0]
    # onnx.save refuses to overwrite an existing external-data file; the
    # prompted-encoder export already wrote one, so remove it first.
    Path(external_filename + ".data").unlink(missing_ok=True)
    onnx.save(
        model,
        filename,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_filename + ".data",
    )


def _to_plain_container(obj: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass

    return obj


def _normalize_prompt_dictionary(obj: Any) -> Dict[str, int]:
    obj = _to_plain_container(obj)
    if not isinstance(obj, dict):
        raise TypeError(type(obj))

    return {str(k): int(v) for k, v in obj.items()}


def _get_config_value(obj: Any, key: str) -> Any:
    if obj is None:
        return None

    try:
        return getattr(obj, key)
    except (AttributeError, KeyError):
        pass

    obj = _to_plain_container(obj)
    if isinstance(obj, dict):
        return obj.get(key)

    return None


def get_prompt_dictionary(asr_model) -> Dict[str, int]:
    """Return the model's language prompt dictionary from NeMo artifacts."""
    cfg = getattr(asr_model, "cfg", None)
    model_defaults = _get_config_value(cfg, "model_defaults")
    if model_defaults is None:
        raise RuntimeError("Could not find cfg.model_defaults in the NeMo model")

    prompt_dictionary = _get_config_value(model_defaults, "prompt_dictionary")
    if prompt_dictionary is None:
        raise RuntimeError(
            "Could not find cfg.model_defaults.prompt_dictionary in the NeMo model"
        )

    try:
        ans = _normalize_prompt_dictionary(prompt_dictionary)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            "cfg.model_defaults.prompt_dictionary must map language strings "
            "to integer prompt ids"
        ) from e

    num_prompts = int(asr_model.num_prompts)
    for language, prompt_id in ans.items():
        if not 0 <= prompt_id < num_prompts:
            raise ValueError(
                "cfg.model_defaults.prompt_dictionary has out-of-range "
                f"prompt id for '{language}': {prompt_id}; expected "
                f"0 <= id < {num_prompts}"
            )

    auto_prompt_id = ans.get("auto")
    if auto_prompt_id != 101:
        raise ValueError(f"Expected auto prompt id 101, got {auto_prompt_id}")

    # The dictionary may use locale-style keys such as en-US or ja-JP; the
    # runtime derives base-code aliases, so accept either form here.
    for language in ["en", "ja"]:
        if not any(k == language or k.startswith(f"{language}-") for k in ans):
            raise RuntimeError(
                "cfg.model_defaults.prompt_dictionary is missing " f"'{language}'"
            )

    return ans


def _find_sentencepiece_processor(obj: Any, max_depth: int = 5) -> Optional[Any]:
    seen = set()

    def is_sentencepiece_processor(value: Any) -> bool:
        return callable(getattr(value, "get_piece_size", None)) and callable(
            getattr(value, "id_to_piece", None)
        )

    def visit(value: Any, depth: int) -> Optional[Any]:
        if value is None or depth > max_depth:
            return None

        if is_sentencepiece_processor(value):
            return value

        obj_id = id(value)
        if obj_id in seen:
            return None
        seen.add(obj_id)

        for name in ["tokenizer", "sp_model", "model", "processor"]:
            if hasattr(value, name):
                found = visit(getattr(value, name), depth + 1)
                if found is not None:
                    return found

        if isinstance(value, dict):
            for v in value.values():
                found = visit(v, depth + 1)
                if found is not None:
                    return found

        return None

    return visit(obj, 0)


def save_tokens(asr_model, filename: str = "tokens.txt") -> int:
    sp = _find_sentencepiece_processor(getattr(asr_model, "tokenizer", None))
    if sp is None:
        raise RuntimeError("Could not find the SentencePiece tokenizer in the model")

    vocab_size = sp.get_piece_size()
    with open(filename, "w", encoding="utf-8") as f:
        for i in range(vocab_size):
            f.write(f"{sp.id_to_piece(i)} {i}\n")
        f.write(f"<blk> {vocab_size}\n")

    print(f"Saved {filename}")
    return vocab_size


def assert_forward_for_export_signature(encoder):
    if not hasattr(encoder, "forward_for_export"):
        raise RuntimeError("Expected encoder.forward_for_export for ONNX export")

    signature = inspect.signature(encoder.forward_for_export)
    missing = [
        name for name in FORWARD_FOR_EXPORT_ARGS if name not in signature.parameters
    ]
    if missing:
        raise RuntimeError(
            "encoder.forward_for_export is missing expected argument(s): "
            f"{missing}. Signature: {signature}"
        )


class PromptedStreamingEncoder(torch.nn.Module):
    def __init__(self, asr_model):
        super().__init__()

        for attr in ["encoder", "prompt_kernel", "num_prompts"]:
            if not hasattr(asr_model, attr):
                raise RuntimeError(
                    "Expected a prompt-conditioned NeMo model with " f"'{attr}'"
                )

        self.encoder = asr_model.encoder
        assert_forward_for_export_signature(self.encoder)

        self.prompt_kernel = asr_model.prompt_kernel
        self.num_prompts = int(asr_model.num_prompts)

    def forward(
        self,
        audio_signal,
        length,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
        prompt_index,
    ):
        encoded, encoded_len, channel_next, time_next, channel_len_next = (
            self.encoder.forward_for_export(
                audio_signal=audio_signal,
                length=length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
            )
        )

        # Mirror NeMo PromptStreamingMixin._apply_prompt_to_encoded(), but make
        # the prompt id a real ONNX input instead of a Python inference setting.
        out_dtype = encoded.dtype
        encoded = encoded.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        batch_size, time_steps, _ = encoded.shape

        prompt = torch.zeros(
            batch_size,
            time_steps,
            self.num_prompts,
            dtype=encoded.dtype,
            device=encoded.device,
        )
        prompt.scatter_(
            2,
            prompt_index.view(batch_size, 1, 1).expand(-1, time_steps, -1),
            1.0,
        )

        encoded = self.prompt_kernel(torch.cat([encoded, prompt], dim=-1)).to(out_dtype)
        encoded = encoded.transpose(1, 2)  # (B, T, D) -> (B, D, T)

        return encoded, encoded_len, channel_next, time_next, channel_len_next


def remove_export_scratch_files():
    patterns = [
        "Constant_*_attr__value",
        "onnx__MatMul_*",
        "layers.*.conv*",
        "pre_encode.conv.*.weight",
        "encoder.export.onnx*",
    ]
    for pattern in patterns:
        for p in Path(".").glob(pattern):
            p.unlink()


def assert_encoder_graph(filename: str):
    model = onnx.load(filename, load_external_data=False)

    input_names = [i.name for i in model.graph.input]
    if input_names != ENCODER_INPUT_NAMES:
        raise RuntimeError(
            f"{filename}: expected encoder inputs {ENCODER_INPUT_NAMES}, "
            f"got {input_names}"
        )

    output_names = [o.name for o in model.graph.output]
    if output_names != ENCODER_OUTPUT_NAMES:
        raise RuntimeError(
            f"{filename}: expected encoder outputs {ENCODER_OUTPUT_NAMES}, "
            f"got {output_names}"
        )


def _module_device_and_dtype(module):
    try:
        p = next(module.parameters())
        return p.device, p.dtype
    except StopIteration:
        return torch.device("cpu"), torch.float32


def export_prompted_encoder(
    asr_model,
    window_size: int,
    cache_last_channel_dim1: int,
    cache_last_channel_dim2: int,
    cache_last_channel_dim3: int,
    cache_last_time_dim1: int,
    cache_last_time_dim2: int,
    cache_last_time_dim3: int,
    auto_prompt_id: int,
):
    device, dtype = _module_device_and_dtype(asr_model.encoder)

    audio_signal = torch.zeros(1, 128, window_size, dtype=dtype, device=device)
    length = torch.full((1,), window_size, dtype=torch.int64, device=device)
    cache_last_channel = torch.zeros(
        1,
        cache_last_channel_dim1,
        cache_last_channel_dim2,
        cache_last_channel_dim3,
        dtype=dtype,
        device=device,
    )
    cache_last_time = torch.zeros(
        1,
        cache_last_time_dim1,
        cache_last_time_dim2,
        cache_last_time_dim3,
        dtype=dtype,
        device=device,
    )
    cache_last_channel_len = torch.zeros(1, dtype=torch.int64, device=device)
    prompt_index = torch.full((1,), auto_prompt_id, dtype=torch.int64, device=device)

    encoder = PromptedStreamingEncoder(asr_model).eval()

    export_kwargs = {}
    export_signature = inspect.signature(torch.onnx.export)
    if "dynamo" in export_signature.parameters:
        export_kwargs["dynamo"] = False
    if "external_data" in export_signature.parameters:
        export_kwargs["external_data"] = True

    torch.onnx.export(
        encoder,
        (
            audio_signal,
            length,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            prompt_index,
        ),
        "encoder.export.onnx",
        input_names=ENCODER_INPUT_NAMES,
        output_names=ENCODER_OUTPUT_NAMES,
        opset_version=17,
        dynamic_axes={
            "audio_signal": {0: "batch", 2: "time"},
            "length": {0: "batch"},
            "cache_last_channel": {0: "batch", 2: "cache_channel_time"},
            "cache_last_time": {0: "batch", 3: "cache_time_width"},
            "cache_last_channel_len": {0: "batch"},
            "prompt_index": {0: "batch"},
            "outputs": {0: "batch", 2: "time"},
            "encoded_lengths": {0: "batch"},
            "cache_last_channel_next": {0: "batch", 2: "cache_channel_time"},
            "cache_last_time_next": {0: "batch", 3: "cache_time_width"},
            "cache_last_channel_next_len": {0: "batch"},
        },
        **export_kwargs,
    )

    model = onnx.load("encoder.export.onnx", load_external_data=True)
    onnx.save_model(
        model,
        "encoder.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="encoder.data",
        size_threshold=0,
    )
    assert_encoder_graph("encoder.onnx")
    for p in Path(".").glob("encoder.export.onnx*"):
        p.unlink()


@torch.no_grad()
def main():
    model_name = "nvidia/nemotron-3.5-asr-streaming-0.6b"

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    vocab_size = save_tokens(asr_model)
    if vocab_size != asr_model.decoder.vocab_size:
        raise ValueError(
            f"SentencePiece vocab size {vocab_size} != decoder vocab size "
            f"{asr_model.decoder.vocab_size}"
        )

    prompt_dictionary = get_prompt_dictionary(asr_model)
    auto_prompt_id = prompt_dictionary["auto"]
    if auto_prompt_id != 101:
        raise ValueError(f"Expected auto prompt id 101, got {auto_prompt_id}")

    asr_model.eval()

    assert asr_model.encoder.streaming_cfg is not None
    print("streaming_cfg", asr_model.encoder.streaming_cfg)
    print("prompt_dictionary", prompt_dictionary)

    chunk_size_ms_list = [80, 160, 320, 560, 1120]
    for ms in chunk_size_ms_list:
        chunk_size = ms // 80 - 1
        print("chunk_size", chunk_size)
        asr_model.encoder.set_default_att_context_size([56, chunk_size])

        print("streaming_cfg", asr_model.encoder.streaming_cfg)
        print("att_context_size", asr_model.encoder.att_context_size)
        print(
            "pre_encode_cache_size",
            asr_model.encoder.streaming_cfg.pre_encode_cache_size,
        )

        if isinstance(asr_model.encoder.streaming_cfg.pre_encode_cache_size, list):
            pre_encode_cache_size = (
                asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
            )
        else:
            pre_encode_cache_size = (
                asr_model.encoder.streaming_cfg.pre_encode_cache_size
            )

        if isinstance(asr_model.encoder.streaming_cfg.chunk_size, list):
            chunk_size = asr_model.encoder.streaming_cfg.chunk_size[1]
        else:
            chunk_size = asr_model.encoder.streaming_cfg.chunk_size

        window_size = chunk_size + pre_encode_cache_size

        print("chunk_size", chunk_size)
        print("pre_encode_cache_size", pre_encode_cache_size)
        print("window_size", window_size)

        chunk_shift = chunk_size

        # cache_last_channel: (batch_size, dim1, dim2, dim3)
        cache_last_channel_dim1 = len(asr_model.encoder.layers)
        cache_last_channel_dim2 = (
            asr_model.encoder.streaming_cfg.last_channel_cache_size
        )
        cache_last_channel_dim3 = asr_model.encoder.d_model

        # cache_last_time: (batch_size, dim1, dim2, dim3)
        cache_last_time_dim1 = len(asr_model.encoder.layers)
        cache_last_time_dim2 = asr_model.encoder.d_model
        cache_last_time_dim3 = asr_model.encoder.conv_context_size[0]

        asr_model.set_export_config({"cache_support": True})

        export_prompted_encoder(
            asr_model=asr_model,
            window_size=window_size,
            cache_last_channel_dim1=cache_last_channel_dim1,
            cache_last_channel_dim2=cache_last_channel_dim2,
            cache_last_channel_dim3=cache_last_channel_dim3,
            cache_last_time_dim1=cache_last_time_dim1,
            cache_last_time_dim2=cache_last_time_dim2,
            cache_last_time_dim3=cache_last_time_dim3,
            auto_prompt_id=auto_prompt_id,
        )
        asr_model.decoder.export("decoder.onnx")
        asr_model.joint.export("joiner.onnx")

        normalize_type = asr_model.cfg.preprocessor.normalize
        if normalize_type == "NA":
            normalize_type = ""

        meta_data = {
            "vocab_size": asr_model.decoder.vocab_size,
            "window_size": window_size,
            "chunk_size_ms": ms,
            "chunk_shift": chunk_shift,
            "normalize_type": normalize_type,
            "cache_last_channel_dim1": cache_last_channel_dim1,
            "cache_last_channel_dim2": cache_last_channel_dim2,
            "cache_last_channel_dim3": cache_last_channel_dim3,
            "cache_last_time_dim1": cache_last_time_dim1,
            "cache_last_time_dim2": cache_last_time_dim2,
            "cache_last_time_dim3": cache_last_time_dim3,
            "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
            "pred_hidden": asr_model.decoder.pred_hidden,
            "subsampling_factor": 8,
            "feat_dim": 128,
            "model_type": type(asr_model).__name__,
            "version": "1",
            "model_author": "NeMo",
            "url": f"https://huggingface.co/{model_name}",
            "comment": "Only the transducer branch is exported",
            "prompt_dictionary": json.dumps(prompt_dictionary, sort_keys=True),
            "auto_prompt_id": auto_prompt_id,
        }
        print("meta_data", meta_data)
        add_meta_data("encoder.onnx", meta_data)
        assert_encoder_graph("encoder.onnx")

        for m in ["encoder", "decoder", "joiner"]:
            quantize_dynamic(
                model_input=f"{m}.onnx",
                model_output=f"{m}.int8.onnx",
                weight_type=QuantType.QUInt8,
            )
        assert_encoder_graph("encoder.int8.onnx")

        Path(str(ms)).mkdir(exist_ok=True)
        for suffix in ["onnx", "data"]:
            for p in Path(".").glob(f"*.{suffix}"):
                p.rename(Path(str(ms)) / p.name)

        print(meta_data)
        print(f"Saved exported models to {ms}")
        remove_export_scratch_files()
        os.system(f"ls -lh {ms}")


if __name__ == "__main__":
    main()
