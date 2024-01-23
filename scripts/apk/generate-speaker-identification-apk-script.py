#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from typing import List, Optional

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
class SpeakerIdentificationModel:
    model_name: str
    short_name: str = ""
    lang: str = ""
    framework: str = ""


def get_3dspeaker_models() -> List[SpeakerIdentificationModel]:
    models = [
        SpeakerIdentificationModel(model_name="3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"),
        SpeakerIdentificationModel(model_name="3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"),
        SpeakerIdentificationModel(model_name="3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx"),
        SpeakerIdentificationModel(model_name="3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"),
        SpeakerIdentificationModel(model_name="3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"),
        SpeakerIdentificationModel(model_name="3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"),
        SpeakerIdentificationModel(model_name="3dspeaker_speech_eres2net_sv_zh-cn_16k-common.onnx"),
    ]

    prefix = '3dspeaker_speech_'
    num = len(prefix)
    for m in models:
        m.framework = '3dspeaker'
        m.short_name = m.model_name[num:-5]
        if '_zh-cn_' in m.model_name:
            m.lang = 'zh'
        elif '_en_' in m.model_name:
            m.lang = 'en'
        else:
            raise ValueError(m)
    return models

def get_wespeaker_models() -> List[SpeakerIdentificationModel]:
    models = [
        SpeakerIdentificationModel(model_name="wespeaker_en_voxceleb_CAM++.onnx"),
        SpeakerIdentificationModel(model_name="wespeaker_en_voxceleb_CAM++_LM.onnx"),
        SpeakerIdentificationModel(model_name="wespeaker_en_voxceleb_resnet152_LM.onnx"),
        SpeakerIdentificationModel(model_name="wespeaker_en_voxceleb_resnet221_LM.onnx"),
        SpeakerIdentificationModel(model_name="wespeaker_en_voxceleb_resnet293_LM.onnx"),
        SpeakerIdentificationModel(model_name="wespeaker_en_voxceleb_resnet34.onnx"),
        SpeakerIdentificationModel(model_name="wespeaker_en_voxceleb_resnet34_LM.onnx"),
        SpeakerIdentificationModel(model_name="wespeaker_zh_cnceleb_resnet34.onnx"),
        SpeakerIdentificationModel(model_name="wespeaker_zh_cnceleb_resnet34_LM.onnx"),
    ]

    prefix = 'wespeaker_xx_'
    num = len(prefix)
    for m in models:
        m.framework = 'wespeaker'
        m.short_name = m.model_name[num:-5]
        if '_zh_' in m.model_name:
            m.lang = 'zh'
        elif '_en_' in m.model_name:
            m.lang = 'en'
        else:
            raise ValueError(m)
    return models

def get_nemo_models() -> List[SpeakerIdentificationModel]:
    models = [
        SpeakerIdentificationModel(model_name="nemo_en_speakerverification_speakernet.onnx"),
        SpeakerIdentificationModel(model_name="nemo_en_titanet_large.onnx"),
        SpeakerIdentificationModel(model_name="nemo_en_titanet_small.onnx"),
    ]

    prefix = 'nemo_en_'
    num = len(prefix)
    for m in models:
        m.framework = 'nemo'
        m.short_name = m.model_name[num:-5]
        if '_zh_' in m.model_name:
            m.lang = 'zh'
        elif '_en_' in m.model_name:
            m.lang = 'en'
        else:
            raise ValueError(m)
    return models



def main():
    args = get_args()
    index = args.index
    total = args.total
    assert 0 <= index < total, (index, total)

    all_model_list = get_3dspeaker_models()
    all_model_list += get_wespeaker_models()
    all_model_list += get_nemo_models()

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

    filename_list = ["./build-apk-speaker-identification.sh"]
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
