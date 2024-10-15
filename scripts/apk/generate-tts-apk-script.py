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
    rule_fars: Optional[List[str]] = None
    data_dir: Optional[str] = None
    dict_dir: Optional[str] = None
    is_char: bool = False
    lang_iso_639_3: str = ""


def convert_lang_to_iso_639_3(models: List[TtsModel]):
    for m in models:
        if m.lang_iso_639_3 == "":
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

    character_models = [
        TtsModel(model_dir="vits-coqui-bg-cv", lang="bg"),
        TtsModel(model_dir="vits-coqui-bn-custom_female", lang="bn"),
        TtsModel(model_dir="vits-coqui-cs-cv", lang="cs"),
        TtsModel(model_dir="vits-coqui-da-cv", lang="da"),
        TtsModel(model_dir="vits-coqui-de-css10", lang="de"),
        TtsModel(model_dir="vits-coqui-es-css10", lang="es"),
        TtsModel(model_dir="vits-coqui-et-cv", lang="et"),
        TtsModel(model_dir="vits-coqui-fi-css10", lang="fi"),
        TtsModel(model_dir="vits-coqui-fr-css10", lang="fr"),
        TtsModel(model_dir="vits-coqui-ga-cv", lang="ga"),
        TtsModel(model_dir="vits-coqui-hr-cv", lang="hr"),
        TtsModel(model_dir="vits-coqui-lt-cv", lang="lt"),
        TtsModel(model_dir="vits-coqui-lv-cv", lang="lv"),
        TtsModel(model_dir="vits-coqui-mt-cv", lang="mt"),
        TtsModel(model_dir="vits-coqui-nl-css10", lang="nl"),
        TtsModel(model_dir="vits-coqui-pl-mai_female", lang="pl"),
        TtsModel(model_dir="vits-coqui-pt-cv", lang="pt"),
        TtsModel(model_dir="vits-coqui-ro-cv", lang="ro"),
        TtsModel(model_dir="vits-coqui-sk-cv", lang="sk"),
        TtsModel(model_dir="vits-coqui-sl-cv", lang="sl"),
        TtsModel(model_dir="vits-coqui-sv-cv", lang="sv"),
        TtsModel(model_dir="vits-coqui-uk-mai", lang="uk"),
    ]
    for m in character_models:
        m.is_char = True
        m.model_name = "model.onnx"

    return models + character_models


def get_piper_models() -> List[TtsModel]:
    models = [
        #  TtsModel(model_dir="vits-piper-es_ES-mls_10246-low"),
        #  TtsModel(model_dir="vits-piper-es_ES-mls_9972-low"),
        #  TtsModel(model_dir="vits-piper-pl_PL-mls_6892-low"),
        TtsModel(model_dir="vits-piper-ar_JO-kareem-low"),
        TtsModel(model_dir="vits-piper-ar_JO-kareem-medium"),
        TtsModel(model_dir="vits-piper-ca_ES-upc_ona-medium"),
        TtsModel(model_dir="vits-piper-ca_ES-upc_ona-x_low"),
        TtsModel(model_dir="vits-piper-ca_ES-upc_pau-x_low"),
        TtsModel(model_dir="vits-piper-ca_ES-upc_pau-x_low"),
        TtsModel(model_dir="vits-piper-cs_CZ-jirka-medium"),
        TtsModel(model_dir="vits-piper-cy_GB-gwryw_gogleddol-medium"),
        TtsModel(model_dir="vits-piper-da_DK-talesyntese-medium"),
        TtsModel(model_dir="vits-piper-de_DE-eva_k-x_low"),
        TtsModel(model_dir="vits-piper-de_DE-karlsson-low"),
        TtsModel(model_dir="vits-piper-de_DE-kerstin-low"),
        #  TtsModel(model_dir="vits-piper-de_DE-mls-medium"),
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
        TtsModel(model_dir="vits-piper-en_GB-aru-medium"),
        TtsModel(model_dir="vits-piper-en_GB-cori-high"),
        TtsModel(model_dir="vits-piper-en_GB-cori-medium"),
        TtsModel(model_dir="vits-piper-en_GB-jenny_dioco-medium"),
        TtsModel(model_dir="vits-piper-en_GB-northern_english_male-medium"),
        TtsModel(model_dir="vits-piper-en_GB-semaine-medium"),
        TtsModel(model_dir="vits-piper-en_GB-southern_english_female-low"),
        TtsModel(model_dir="vits-piper-en_GB-southern_english_female-medium"),
        TtsModel(model_dir="vits-piper-en_GB-southern_english_male-medium"),
        TtsModel(model_dir="vits-piper-en_GB-sweetbbak-amy"),
        TtsModel(model_dir="vits-piper-en_GB-vctk-medium"),
        TtsModel(model_dir="vits-piper-en_US-amy-low"),
        TtsModel(model_dir="vits-piper-en_US-amy-medium"),
        TtsModel(model_dir="vits-piper-en_US-arctic-medium"),
        TtsModel(model_dir="vits-piper-en_US-bryce-medium"),
        TtsModel(model_dir="vits-piper-en_US-danny-low"),
        TtsModel(model_dir="vits-piper-en_US-glados"),
        TtsModel(model_dir="vits-piper-en_US-hfc_female-medium"),
        TtsModel(model_dir="vits-piper-en_US-hfc_male-medium"),
        TtsModel(model_dir="vits-piper-en_US-joe-medium"),
        TtsModel(model_dir="vits-piper-en_US-john-medium"),
        TtsModel(model_dir="vits-piper-en_US-kathleen-low"),
        TtsModel(model_dir="vits-piper-en_US-kristin-medium"),
        TtsModel(model_dir="vits-piper-en_US-kusal-medium"),
        TtsModel(model_dir="vits-piper-en_US-l2arctic-medium"),
        TtsModel(model_dir="vits-piper-en_US-lessac-high"),
        TtsModel(model_dir="vits-piper-en_US-lessac-low"),
        TtsModel(model_dir="vits-piper-en_US-lessac-medium"),
        TtsModel(model_dir="vits-piper-en_US-libritts-high"),
        TtsModel(model_dir="vits-piper-en_US-libritts_r-medium"),
        TtsModel(model_dir="vits-piper-en_US-ljspeech-high"),
        TtsModel(model_dir="vits-piper-en_US-ljspeech-medium"),
        TtsModel(model_dir="vits-piper-en_US-norman-medium"),
        TtsModel(model_dir="vits-piper-en_US-ryan-high"),
        TtsModel(model_dir="vits-piper-en_US-ryan-low"),
        TtsModel(model_dir="vits-piper-en_US-ryan-medium"),
        TtsModel(model_dir="vits-piper-es-glados-medium"),
        TtsModel(model_dir="vits-piper-es_ES-carlfm-x_low"),
        TtsModel(model_dir="vits-piper-es_ES-davefx-medium"),
        TtsModel(model_dir="vits-piper-es_ES-sharvard-medium"),
        TtsModel(model_dir="vits-piper-es_MX-ald-medium"),
        TtsModel(model_dir="vits-piper-es_MX-claude-high"),
        TtsModel(model_dir="vits-piper-fa_IR-amir-medium"),
        TtsModel(model_dir="vits-piper-fa_IR-gyro-medium"),
        TtsModel(model_dir="vits-piper-fi_FI-harri-low"),
        TtsModel(model_dir="vits-piper-fi_FI-harri-medium"),
        #  TtsModel(model_dir="vits-piper-fr_FR-mls-medium"),
        TtsModel(model_dir="vits-piper-fr_FR-siwis-low"),
        TtsModel(model_dir="vits-piper-fr_FR-siwis-medium"),
        TtsModel(model_dir="vits-piper-fr_FR-tom-medium"),
        TtsModel(model_dir="vits-piper-fr_FR-upmc-medium"),
        TtsModel(model_dir="vits-piper-hu_HU-anna-medium"),
        TtsModel(model_dir="vits-piper-hu_HU-berta-medium"),
        TtsModel(model_dir="vits-piper-hu_HU-imre-medium"),
        TtsModel(model_dir="vits-piper-is_IS-bui-medium"),
        TtsModel(model_dir="vits-piper-is_IS-salka-medium"),
        TtsModel(model_dir="vits-piper-is_IS-steinn-medium"),
        TtsModel(model_dir="vits-piper-is_IS-ugla-medium"),
        TtsModel(model_dir="vits-piper-it_IT-paola-medium"),
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
        #  TtsModel(model_dir="vits-piper-nl_NL-mls-medium"),
        #  TtsModel(model_dir="vits-piper-nl_NL-mls_5809-low"),
        #  TtsModel(model_dir="vits-piper-nl_NL-mls_7432-low"),
        TtsModel(model_dir="vits-piper-no_NO-talesyntese-medium"),
        TtsModel(model_dir="vits-piper-pl_PL-darkman-medium"),
        TtsModel(model_dir="vits-piper-pl_PL-gosia-medium"),
        TtsModel(model_dir="vits-piper-pl_PL-mc_speech-medium"),
        TtsModel(model_dir="vits-piper-pt_BR-edresson-low"),
        TtsModel(model_dir="vits-piper-pt_BR-faber-medium"),
        TtsModel(model_dir="vits-piper-pt_PT-tugao-medium"),
        TtsModel(model_dir="vits-piper-ro_RO-mihai-medium"),
        TtsModel(model_dir="vits-piper-ru_RU-denis-medium"),
        TtsModel(model_dir="vits-piper-ru_RU-dmitri-medium"),
        TtsModel(model_dir="vits-piper-ru_RU-irina-medium"),
        TtsModel(model_dir="vits-piper-ru_RU-ruslan-medium"),
        TtsModel(model_dir="vits-piper-sk_SK-lili-medium"),
        TtsModel(model_dir="vits-piper-sl_SI-artur-medium"),
        TtsModel(model_dir="vits-piper-sr_RS-serbski_institut-medium"),
        TtsModel(model_dir="vits-piper-sv_SE-nst-medium"),
        TtsModel(model_dir="vits-piper-sw_CD-lanfrica-medium"),
        TtsModel(model_dir="vits-piper-tr_TR-dfki-medium"),
        TtsModel(model_dir="vits-piper-tr_TR-fahrettin-medium"),
        TtsModel(model_dir="vits-piper-tr_TR-fettah-medium"),
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
    chinese_models = [
        # Chinese
        TtsModel(
            model_dir="vits-icefall-zh-aishell3",
            model_name="model.onnx",
            lang="zh",
            rule_fsts="vits-icefall-zh-aishell3/phone.fst,vits-icefall-zh-aishell3/date.fst,vits-icefall-zh-aishell3/number.fst,vits-icefall-zh-aishell3/new_heteronym.fst",
            rule_fars="vits-icefall-zh-aishell3/rule.far",
        ),
        TtsModel(
            model_dir="vits-zh-aishell3",
            model_name="vits-aishell3.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-zh-hf-doom",
            model_name="doom.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-zh-hf-echo",
            model_name="echo.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-zh-hf-zenyatta",
            model_name="zenyatta.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-zh-hf-abyssinvoker",
            model_name="abyssinvoker.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-zh-hf-keqing",
            model_name="keqing.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-zh-hf-eula",
            model_name="eula.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-zh-hf-bronya",
            model_name="bronya.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-zh-hf-theresa",
            model_name="theresa.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-zh-hf-fanchen-wnj",
            model_name="vits-zh-hf-fanchen-wnj.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-melo-tts-zh_en",
            model_name="model.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-zh-hf-fanchen-C",
            model_name="vits-zh-hf-fanchen-C.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-zh-hf-fanchen-ZhiHuiLaoZhe",
            model_name="vits-zh-hf-fanchen-ZhiHuiLaoZhe.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-zh-hf-fanchen-ZhiHuiLaoZhe_new",
            model_name="vits-zh-hf-fanchen-ZhiHuiLaoZhe_new.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="vits-zh-hf-fanchen-unity",
            model_name="vits-zh-hf-fanchen-unity.onnx",
            lang="zh",
        ),
        TtsModel(
            model_dir="sherpa-onnx-vits-zh-ll",
            model_name="model.onnx",
            lang="zh",
        ),
    ]

    rule_fsts = ["phone.fst", "date.fst", "number.fst"]
    for m in chinese_models:
        s = [f"{m.model_dir}/{r}" for r in rule_fsts]
        if (
            "vits-zh-hf" in m.model_dir
            or "sherpa-onnx-vits-zh-ll" == m.model_dir
            or "melo-tts" in m.model_dir
        ):
            s = s[:-1]
            m.dict_dir = m.model_dir + "/dict"
        else:
            m.rule_fars = f"{m.model_dir}/rule.far"

        m.rule_fsts = ",".join(s)

    all_models = chinese_models + [
        TtsModel(
            model_dir="vits-cantonese-hf-xiaomaiiwn",
            model_name="vits-cantonese-hf-xiaomaiiwn.onnx",
            lang="cantonese",
            lang_iso_639_3="yue",
            rule_fsts="vits-cantonese-hf-xiaomaiiwn/rule.fst",
        ),
        # English (US)
        TtsModel(model_dir="vits-vctk", model_name="vits-vctk.onnx", lang="en"),
        #  TtsModel(model_dir="vits-ljs", model_name="vits-ljs.onnx", lang="en"),
        # fmt: on
    ]

    return all_models


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
