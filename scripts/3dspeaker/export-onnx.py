#!/usr/bin/env python3
# Copyright      2023-2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import json
import os
import pathlib
import re
from typing import Dict

import onnx
import torch
from infer_sv import supports
from modelscope.hub.snapshot_download import snapshot_download
from speakerlab.utils.builder import dynamic_import


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "speech_campplus_sv_en_voxceleb_16k",
            "speech_campplus_sv_zh-cn_16k-common",
            "speech_eres2net_sv_en_voxceleb_16k",
            "speech_eres2net_sv_zh-cn_16k-common",
            "speech_eres2net_base_200k_sv_zh-cn_16k-common",
            "speech_eres2net_base_sv_zh-cn_3dspeaker_16k",
            "speech_eres2net_large_sv_zh-cn_3dspeaker_16k",
        ],
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    local_model_dir = "pretrained"
    model_id = f"damo/{args.model}"
    conf = supports[model_id]
    cache_dir = snapshot_download(
        model_id,
        revision=conf["revision"],
    )
    cache_dir = pathlib.Path(cache_dir)

    save_dir = os.path.join(local_model_dir, model_id.split("/")[1])
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    download_files = ["examples", conf["model_pt"]]
    for src in cache_dir.glob("*"):
        if re.search("|".join(download_files), src.name):
            dst = save_dir / src.name
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            dst.symlink_to(src)
    pretrained_model = save_dir / conf["model_pt"]
    pretrained_state = torch.load(pretrained_model, map_location="cpu")

    model = conf["model"]
    embedding_model = dynamic_import(model["obj"])(**model["args"])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()

    with open(f"{cache_dir}/configuration.json") as f:
        json_config = json.loads(f.read())
        print(json_config)

    T = 100
    C = 80
    x = torch.rand(1, T, C)
    filename = f"{args.model}.onnx"
    torch.onnx.export(
        embedding_model,
        x,
        filename,
        opset_version=13,
        input_names=["x"],
        output_names=["embedding"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "embeddings": {0: "N"},
        },
    )

    # all models from 3d-speaker expect input samples in the range
    # [-1, 1]
    normalize_samples = 1

    # all models from 3d-speaker normalize the features by the global mean
    feature_normalize_type = "global-mean"
    sample_rate = json_config["model"]["model_config"]["sample_rate"]

    feat_dim = conf["model"]["args"]["feat_dim"]
    assert feat_dim == 80, feat_dim

    output_dim = conf["model"]["args"]["embedding_size"]

    if "zh-cn" in args.model:
        language = "Chinese"
    elif "en" in args.model:
        language = "English"
    else:
        raise ValueError(f"Unsupported language for model {args.model}")

    comment = f"This model is from damo/{args.model}"
    url = f"https://www.modelscope.cn/models/damo/{args.model}/summary"

    meta_data = {
        "framework": "3d-speaker",
        "language": language,
        "url": url,
        "comment": comment,
        "sample_rate": sample_rate,
        "output_dim": output_dim,
        "normalize_samples": normalize_samples,
        "feature_normalize_type": feature_normalize_type,
    }
    print(meta_data)
    add_meta_data(filename=filename, meta_data=meta_data)


main()
