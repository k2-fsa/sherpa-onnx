#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
from typing import Dict

import nemo.collections.asr as nemo_asr
import onnx
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "speakerverification_speakernet",
            "titanet_large",
            "titanet_small",
            "ecapa_tdnn",
        ],
    )
    return parser.parse_args()


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


@torch.no_grad()
def main():
    args = get_args()
    speaker_model_config = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name=args.model, return_config=True
    )
    preprocessor_config = speaker_model_config["preprocessor"]

    print(args.model)
    print(speaker_model_config)
    print(preprocessor_config)

    assert preprocessor_config["n_fft"] == 512, preprocessor_config

    assert (
        preprocessor_config["_target_"]
        == "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor"
    ), preprocessor_config

    assert preprocessor_config["frame_splicing"] == 1, preprocessor_config

    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name=args.model
    )
    speaker_model.eval()
    filename = f"nemo_en_{args.model}.onnx"
    speaker_model.export(filename)

    print(f"Adding metadata to {filename}")

    comment = "This model is from NeMo."
    url = {
        "titanet_large": "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large",
        "titanet_small": "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small",
        "speakerverification_speakernet": "https://ngc.nvidia.com/catalog/models/nvidia:nemo:speakerverification_speakernet",
        "ecapa_tdnn": "https://ngc.nvidia.com/catalog/models/nvidia:nemo:ecapa_tdnn",
    }[args.model]

    language = "English"

    meta_data = {
        "framework": "nemo",
        "language": language,
        "url": url,
        "comment": comment,
        "sample_rate": preprocessor_config["sample_rate"],
        "output_dim": speaker_model_config["decoder"]["emb_sizes"],
        "feature_normalize_type": preprocessor_config["normalize"],
        "window_size_ms": int(float(preprocessor_config["window_size"]) * 1000),
        "window_stride_ms": int(float(preprocessor_config["window_stride"]) * 1000),
        "window_type": preprocessor_config["window"],  # e.g., hann
        "feat_dim": preprocessor_config["features"],
    }
    print(meta_data)
    add_meta_data(filename=filename, meta_data=meta_data)


if __name__ == "__main__":
    main()
