#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from typing import List

import jinja2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total",
        type=int,
        default=1,
        help="Number of runners",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the current runner",
    )
    return parser.parse_args()


@dataclass
class SpeakerSegmentationModel:
    model_name: str
    short_name: str


@dataclass
class SpeakerEmbeddingModel:
    model_name: str
    short_name: str


@dataclass
class Model:
    segmentation: SpeakerSegmentationModel
    embedding: SpeakerEmbeddingModel


def get_segmentation_models() -> List[SpeakerSegmentationModel]:
    models = [
        SpeakerSegmentationModel(
            model_name="sherpa-onnx-pyannote-segmentation-3-0",
            short_name="pyannote_audio",
        ),
        SpeakerSegmentationModel(
            model_name="sherpa-onnx-reverb-diarization-v1",
            short_name="revai_v1",
        ),
    ]

    return models


def get_embedding_models() -> List[SpeakerEmbeddingModel]:
    models = [
        SpeakerSegmentationModel(
            model_name="3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k",
            short_name="3dspeaker",
        ),
        SpeakerSegmentationModel(
            model_name="nemo_en_titanet_small",
            short_name="nemo",
        ),
    ]
    return models


def main():
    args = get_args()
    index = args.index
    total = args.total
    assert 0 <= index < total, (index, total)

    segmentation_models = get_segmentation_models()
    embedding_models = get_embedding_models()

    all_model_list = []
    for s in segmentation_models:
        for e in embedding_models:
            all_model_list.append(Model(segmentation=s, embedding=e))

    num_models = len(all_model_list)

    num_per_runner = num_models // total
    if num_per_runner <= 0:
        raise ValueError(f"num_models: {num_models}, num_runners: {total}")

    start = index * num_per_runner
    end = start + num_per_runner

    remaining = num_models - args.total * num_per_runner

    print(f"{index}/{total}: {start}-{end}/{num_models}")

    d = dict()
    d["model_list"] = all_model_list[start:end]
    if index < remaining:
        s = args.total * num_per_runner + index
        d["model_list"].append(all_model_list[s])
        print(f"{s}/{num_models}")

    filename_list = ["./build-apk-speaker-diarization.sh"]
    for filename in filename_list:
        environment = jinja2.Environment()
        with open(f"{filename}.in") as f:
            s = f.read()
        template = environment.from_string(s)

        s = template.render(**d)
        with open(filename, "w") as f:
            print(s, file=f)


if __name__ == "__main__":
    main()
