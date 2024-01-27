#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from typing import List, Optional

import jinja2

# pip install iso639-lang
from iso639 import Lang


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
class TtsModel:
    model_dir: str
    model_name: str = ""
    lang: str = ""  # en, zh, fr, de, etc.
    rule_fsts: Optional[List[str]] = None
    data_dir: Optional[str] = None
    lang_iso_639_3: str = ""


def convert_lang_to_iso_639_3(models: List[TtsModel]):
    for m in models:
        m.lang_iso_639_3 = Lang(m.lang).pt3


def get_coqui_models() -> List[TtsModel]:
    # English (coqui-ai/TTS)
    models = [
        TtsModel(model_dir="vits-coqui-en-ljspeech"),
        TtsModel(model_dir="vits-coqui-en-ljspeech-neon"),
        TtsModel(model_dir="vits-coqui-en-vctk"),
        #  TtsModel(model_dir="vits-coqui-en-jenny"),
    ]

    for m in models:
        m.data_dir = m.model_dir + "/" + "espeak-ng-data"
        m.model_name = "model.onnx"
        m.lang = "en"

    return models


def get_piper_models() -> List[TtsModel]:
    models = [
        TtsModel(model_dir="vits-piper-ar_JO-kareem-low"),
        TtsModel(model_dir="vits-piper-ar_JO-kareem-medium"),
        TtsModel(model_dir="vits-piper-ca_ES-upc_ona-medium"),
        TtsModel(model_dir="vits-piper-ca_ES-upc_ona-x_low"),
        TtsModel(model_dir="vits-piper-ca_ES-upc_pau-x_low"),
        TtsModel(model_dir="vits-piper-ca_ES-upc_pau-x_low"),
        TtsModel(model_dir="vits-piper-cs_CZ-jirka-medium"),
        TtsModel(model_dir="vits-piper-da_DK-talesyntese-medium"),
        TtsModel(model_dir="vits-piper-de_DE-eva_k-x_low"),
        TtsModel(model_dir="vits-piper-de_DE-karlsson-low"),
        TtsModel(model_dir="vits-piper-de_DE-kerstin-low"),
        TtsModel(model_dir="vits-piper-de_DE-pavoque-low"),
        TtsModel(model_dir="vits-piper-de_DE-ramona-low"),
        TtsModel(model_dir="vits-piper-de_DE-thorsten-high"),
        TtsModel(model_dir="vits-piper-de_DE-thorsten-low"),
        TtsModel(model_dir="vits-piper-de_DE-thorsten-medium"),
        TtsModel(model_dir="vits-piper-de_DE-thorsten_emotional-medium"),
        TtsModel(model_dir="vits-piper-el_GR-rapunzelina-low"),
        TtsModel(model_dir="vits-piper-en_GB-alan-low"),
        TtsModel(model_dir="vits-piper-en_GB-alan-medium"),
        TtsModel(model_dir="vits-piper-en_GB-alba-medium"),
        TtsModel(model_dir="vits-piper-en_GB-jenny_dioco-medium"),
        TtsModel(model_dir="vits-piper-en_GB-northern_english_male-medium"),
        TtsModel(model_dir="vits-piper-en_GB-semaine-medium"),
        TtsModel(model_dir="vits-piper-en_GB-southern_english_female-low"),
        TtsModel(model_dir="vits-piper-en_GB-sweetbbak-amy"),
        TtsModel(model_dir="vits-piper-en_GB-vctk-medium"),
        TtsModel(model_dir="vits-piper-en_US-glados"),
        TtsModel(model_dir="vits-piper-en_US-amy-low"),
        TtsModel(model_dir="vits-piper-en_US-amy-medium"),
        TtsModel(model_dir="vits-piper-en_US-arctic-medium"),
        TtsModel(model_dir="vits-piper-en_US-danny-low"),
        TtsModel(model_dir="vits-piper-en_US-hfc_male-medium"),
        TtsModel(model_dir="vits-piper-en_US-joe-medium"),
        TtsModel(model_dir="vits-piper-en_US-kathleen-low"),
        TtsModel(model_dir="vits-piper-en_US-kusal-medium"),
        TtsModel(model_dir="vits-piper-en_US-l2arctic-medium"),
        TtsModel(model_dir="vits-piper-en_US-lessac-high"),
        TtsModel(model_dir="vits-piper-en_US-lessac-low"),
        TtsModel(model_dir="vits-piper-en_US-lessac-medium"),
        TtsModel(model_dir="vits-piper-en_US-libritts-high"),
        TtsModel(model_dir="vits-piper-en_US-libritts_r-medium"),
        TtsModel(model_dir="vits-piper-en_US-ryan-high"),
        TtsModel(model_dir="vits-piper-en_US-ryan-low"),
        TtsModel(model_dir="vits-piper-en_US-ryan-medium"),
        TtsModel(model_dir="vits-piper-es-glados-medium"),
        TtsModel(model_dir="vits-piper-es_ES-carlfm-x_low"),
        TtsModel(model_dir="vits-piper-es_ES-davefx-medium"),
        #  TtsModel(model_dir="vits-piper-es_ES-mls_10246-low"),
        #  TtsModel(model_dir="vits-piper-es_ES-mls_9972-low"),
        TtsModel(model_dir="vits-piper-es_ES-sharvard-medium"),
        TtsModel(model_dir="vits-piper-es_MX-ald-medium"),
        TtsModel(model_dir="vits-piper-fa_IR-amir-medium"),
        TtsModel(model_dir="vits-piper-fa_IR-gyro-medium"),
        TtsModel(model_dir="vits-piper-fi_FI-harri-low"),
        TtsModel(model_dir="vits-piper-fi_FI-harri-medium"),
        TtsModel(model_dir="vits-piper-fr_FR-siwis-low"),
        TtsModel(model_dir="vits-piper-fr_FR-siwis-medium"),
        TtsModel(model_dir="vits-piper-fr_FR-upmc-medium"),
        TtsModel(model_dir="vits-piper-hu_HU-anna-medium"),
        TtsModel(model_dir="vits-piper-hu_HU-berta-medium"),
        TtsModel(model_dir="vits-piper-hu_HU-imre-medium"),
        TtsModel(model_dir="vits-piper-is_IS-bui-medium"),
        TtsModel(model_dir="vits-piper-is_IS-salka-medium"),
        TtsModel(model_dir="vits-piper-is_IS-steinn-medium"),
        TtsModel(model_dir="vits-piper-is_IS-ugla-medium"),
        TtsModel(model_dir="vits-piper-it_IT-riccardo-x_low"),
        TtsModel(model_dir="vits-piper-ka_GE-natia-medium"),
        TtsModel(model_dir="vits-piper-kk_KZ-iseke-x_low"),
        TtsModel(model_dir="vits-piper-kk_KZ-issai-high"),
        TtsModel(model_dir="vits-piper-kk_KZ-raya-x_low"),
        TtsModel(model_dir="vits-piper-lb_LU-marylux-medium"),
        TtsModel(model_dir="vits-piper-ne_NP-google-medium"),
        TtsModel(model_dir="vits-piper-ne_NP-google-x_low"),
        TtsModel(model_dir="vits-piper-nl_BE-nathalie-medium"),
        TtsModel(model_dir="vits-piper-nl_BE-nathalie-x_low"),
        TtsModel(model_dir="vits-piper-nl_BE-rdh-medium"),
        TtsModel(model_dir="vits-piper-nl_BE-rdh-x_low"),
        TtsModel(model_dir="vits-piper-nl_NL-mls_5809-low"),
        TtsModel(model_dir="vits-piper-nl_NL-mls_7432-low"),
        TtsModel(model_dir="vits-piper-no_NO-talesyntese-medium"),
        TtsModel(model_dir="vits-piper-pl_PL-darkman-medium"),
        TtsModel(model_dir="vits-piper-pl_PL-gosia-medium"),
        TtsModel(model_dir="vits-piper-pl_PL-mc_speech-medium"),
        #  TtsModel(model_dir="vits-piper-pl_PL-mls_6892-low"),
        TtsModel(model_dir="vits-piper-pt_BR-edresson-low"),
        TtsModel(model_dir="vits-piper-pt_BR-faber-medium"),
        TtsModel(model_dir="vits-piper-pt_PT-tugao-medium"),
        TtsModel(model_dir="vits-piper-ro_RO-mihai-medium"),
        TtsModel(model_dir="vits-piper-ru_RU-denis-medium"),
        TtsModel(model_dir="vits-piper-ru_RU-dmitri-medium"),
        TtsModel(model_dir="vits-piper-ru_RU-irina-medium"),
        TtsModel(model_dir="vits-piper-ru_RU-ruslan-medium"),
        TtsModel(model_dir="vits-piper-sl_SI-artur-medium"),
        TtsModel(model_dir="vits-piper-sk_SK-lili-medium"),
        TtsModel(model_dir="vits-piper-sr_RS-serbski_institut-medium"),
        TtsModel(model_dir="vits-piper-sv_SE-nst-medium"),
        TtsModel(model_dir="vits-piper-sw_CD-lanfrica-medium"),
        TtsModel(model_dir="vits-piper-tr_TR-dfki-medium"),
        TtsModel(model_dir="vits-piper-tr_TR-fahrettin-medium"),
        TtsModel(model_dir="vits-piper-uk_UA-lada-x_low"),
        TtsModel(model_dir="vits-piper-uk_UA-ukrainian_tts-medium"),
        TtsModel(model_dir="vits-piper-vi_VN-25hours_single-low"),
        TtsModel(model_dir="vits-piper-vi_VN-vais1000-medium"),
        TtsModel(model_dir="vits-piper-vi_VN-vivos-x_low"),
        TtsModel(model_dir="vits-piper-zh_CN-huayan-medium"),
    ]

    for m in models:
        m.data_dir = m.model_dir + "/" + "espeak-ng-data"
        m.model_name = m.model_dir[len("vits-piper-") :] + ".onnx"
        m.lang = m.model_dir.split("-")[2][:2]

    return models


def get_mimic3_models() -> List[TtsModel]:
    models = [
        TtsModel(model_dir="vits-mimic3-af_ZA-google-nwu_low"),
        TtsModel(model_dir="vits-mimic3-bn-multi_low"),
        TtsModel(model_dir="vits-mimic3-es_ES-m-ailabs_low"),
        TtsModel(model_dir="vits-mimic3-fa-haaniye_low"),
        TtsModel(model_dir="vits-mimic3-fi_FI-harri-tapani-ylilammi_low"),
        TtsModel(model_dir="vits-mimic3-gu_IN-cmu-indic_low"),
        TtsModel(model_dir="vits-mimic3-hu_HU-diana-majlinger_low"),
        TtsModel(model_dir="vits-mimic3-ko_KO-kss_low"),
        TtsModel(model_dir="vits-mimic3-ne_NP-ne-google_low"),
        TtsModel(model_dir="vits-mimic3-pl_PL-m-ailabs_low"),
        TtsModel(model_dir="vits-mimic3-tn_ZA-google-nwu_low"),
        TtsModel(model_dir="vits-mimic3-vi_VN-vais1000_low"),
    ]
    for m in models:
        m.data_dir = m.model_dir + "/" + "espeak-ng-data"
        m.model_name = m.model_dir[len("vits-mimic3-") :] + ".onnx"
        m.lang = m.model_dir.split("-")[2][:2]

    return models


def get_vits_models() -> List[TtsModel]:
    return [
        # Chinese
        TtsModel(
            model_dir="vits-zh-aishell3",
            model_name="vits-aishell3.onnx",
            lang="zh",
            rule_fsts="vits-zh-aishell3/rule.fst",
        ),
        #  TtsModel(
        #      model_dir="vits-zh-hf-doom",
        #      model_name="doom.onnx",
        #      lang="zh",
        #      rule_fsts="vits-zh-hf-doom/rule.fst",
        #  ),
        #  TtsModel(
        #      model_dir="vits-zh-hf-echo",
        #      model_name="echo.onnx",
        #      lang="zh",
        #      rule_fsts="vits-zh-hf-echo/rule.fst",
        #  ),
        #  TtsModel(
        #      model_dir="vits-zh-hf-zenyatta",
        #      model_name="zenyatta.onnx",
        #      lang="zh",
        #      rule_fsts="vits-zh-hf-zenyatta/rule.fst",
        #  ),
        #  TtsModel(
        #      model_dir="vits-zh-hf-abyssinvoker",
        #      model_name="abyssinvoker.onnx",
        #      lang="zh",
        #      rule_fsts="vits-zh-hf-abyssinvoker/rule.fst",
        #  ),
        #  TtsModel(
        #      model_dir="vits-zh-hf-keqing",
        #      model_name="keqing.onnx",
        #      lang="zh",
        #      rule_fsts="vits-zh-hf-keqing/rule.fst",
        #  ),
        #  TtsModel(
        #      model_dir="vits-zh-hf-eula",
        #      model_name="eula.onnx",
        #      lang="zh",
        #      rule_fsts="vits-zh-hf-eula/rule.fst",
        #  ),
        #  TtsModel(
        #      model_dir="vits-zh-hf-bronya",
        #      model_name="bronya.onnx",
        #      lang="zh",
        #      rule_fsts="vits-zh-hf-bronya/rule.fst",
        #  ),
        #  TtsModel(
        #      model_dir="vits-zh-hf-theresa",
        #      model_name="theresa.onnx",
        #      lang="zh",
        #      rule_fsts="vits-zh-hf-theresa/rule.fst",
        #  ),
        # English (US)
        TtsModel(model_dir="vits-vctk", model_name="vits-vctk.onnx", lang="en"),
        #  TtsModel(model_dir="vits-ljs", model_name="vits-ljs.onnx", lang="en"),
        # fmt: on
    ]


def main():
    args = get_args()
    index = args.index
    total = args.total
    assert 0 <= index < total, (index, total)
    d = dict()

    all_model_list = get_vits_models()
    all_model_list += get_piper_models()
    all_model_list += get_mimic3_models()
    all_model_list += get_coqui_models()
    convert_lang_to_iso_639_3(all_model_list)

    num_models = len(all_model_list)

    num_per_runner = num_models // total
    if num_per_runner <= 0:
        raise ValueError(f"num_models: {num_models}, num_runners: {total}")

    start = index * num_per_runner
    end = start + num_per_runner

    remaining = num_models - args.total * num_per_runner

    print(f"{index}/{total}: {start}-{end}/{num_models}")
    d["tts_model_list"] = all_model_list[start:end]
    if index < remaining:
        s = args.total * num_per_runner + index
        d["tts_model_list"].append(all_model_list[s])
        print(f"{s}/{num_models}")

    filename_list = ["./build-apk-tts.sh", "./build-apk-tts-engine.sh"]
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
