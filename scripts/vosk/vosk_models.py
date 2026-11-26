#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# THE list of vosk models (https://alphacephei.com/vosk/models), shared
# by the upload flows (upload-hf/, upload-onnx/): their generate.py is
# the same driver, importing this module -- so this file is the single
# place to add or edit a model.
#
# Only the ASR models are listed (no speaker-id / text models).  The
# list mirrors the "Vosk Speech Recognition Models" page (the
# authoritative source), plus a few models the page does not list that
# are found in the https://huggingface.co/Derur/vosk-models mirror (see
# the kz entries).  To refresh the list from the page:
#
#   curl -sL https://alphacephei.com/vosk/models \
#     | grep -oE 'href="[^"]*\.zip"' | sed 's/href="//;s/"//' | sort -u
#
# Per-model data on top of the name/url:
#   * license  -- the "License" column of that page ("" if unlisted)
#   * wav, expect -- the per-language test wav (a <lang>.wav from the
#     asr-models GitHub release, see TEST_WAVS) and its transcript; the
#     export flows gate on them

from dataclasses import dataclass


@dataclass
class VoskModel:
    name: str  # e.g. vosk-model-small-cn-0.22 (the dir inside the zip)
    url: str  # https://alphacephei.com/vosk/models/<name>.zip
    kind: str = "asr"
    lang: str = ""  # language code, filled by get_all_models()
    index: int = 0
    license: str = ""  # filled by get_all_models() from LICENSES
    wav: str = ""  # test wav name (e.g. zh.wav), filled from TEST_WAVS
    expect: str = ""  # transcript of the test wav, filled from TEST_WAVS


LANG_NAMES = {
    "ar": "Arabic",
    "br": "Breton",
    "ca": "Catalan",
    "cn": "Chinese",
    "cs": "Czech",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "gu": "Gujarati",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "ko": "Korean",
    "ky": "Kyrgyz",
    "kz": "Kazakh",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "te": "Telugu",
    "tg": "Tajik",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "uz": "Uzbek",
    "vn": "Vietnamese",
}


def lang_of(name: str) -> str:
    """Language code from a model name (vosk-model-small-cn-0.22 -> cn)."""
    parts = name.split("-")
    i = 2
    if parts[i] == "small":
        i += 1
    return parts[i]


# test wavs from the asr-models GitHub release, one per language
# (lang code -> wav name).  These are the languages the export flows
# gate on; models of other languages get no wav/expect.
TEST_WAV_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

TEST_WAVS = {
    # lang: (wav name, its transcript)
    "cn": ("zh.wav", "不要问你的国家能为你做什么，而要问你能为你的国家做什么。"),
    "en": (
        "en.wav",
        "Ask not what your country can do for you, "
        "ask what you can do for your country.",
    ),
    "de": ("de.wav", "Alles hat ein Ende, nur die Wurst hat zwei."),
    "fr": (
        "fr.wav",
        "Ne vous demandez pas ce que votre pays peut faire pour vous, "
        "demandez-vous plutôt ce que vous pouvez faire pour lui.",
    ),
    "ja": (
        "ja.wav",
        "国があなたのために何ができるかを問うのではなく、" "あな たが国のために何ができるかを問いてください。",
    ),
    "ko": (
        "ko.wav",
        "조국 이 당신 을 위해 무엇 을 해 줄 수 있 는지 묻 지 말고, " "당 신이 조 국 을 위해 무엇 을 할 수 있 는지 물으십시오.",
    ),
    "vn": (
        "vi.wav",
        "ĐỪNG HỎI ĐẤT NƯỚC CÓ THỂ LÀM GÌ CHO BẠN "
        "HÃY HỎI BẠN CÓ THỂ LÀM GÌ CHO ĐẤT NƯỚC",
    ),
    "uk": (
        "uk.wav",
        "ти питай що твоя країна може зробити для тебе "
        "папитай що ти можеш зробити для своєї країни",
    ),
}

# model name -> license, from the "License" column of
# https://alphacephei.com/vosk/models ("" = not listed there)
LICENSES = {
    "vosk-model-ar-0.22-linto-1.1.0": "AGPL",
    "vosk-model-ar-mgb2-0.4": "Apache 2.0",
    "vosk-model-ar-tn-0.1-linto": "Apache 2.0",
    "vosk-model-small-ar-tn-0.1-linto": "Apache 2.0",
    "vosk-model-br-0.8": "MIT",
    "vosk-model-small-ca-0.4": "Apache 2.0",
    "vosk-model-cn-0.22": "Apache 2.0",
    "vosk-model-cn-kaldi-multicn-0.15": "Apache 2.0",
    "vosk-model-small-cn-0.22": "Apache 2.0",
    "vosk-model-small-cs-0.4-rhasspy": "MIT",
    "vosk-model-de-0.21": "Apache 2.0",
    "vosk-model-de-tuda-0.6-900k": "Apache 2.0",
    "vosk-model-small-de-0.15": "Apache 2.0",
    "vosk-model-small-de-zamia-0.3": "LGPL-3.0",
    "vosk-model-el-gr-0.7": "Apache 2.0",
    "vosk-model-en-in-0.5": "Apache 2.0",
    "vosk-model-en-us-0.21": "Apache 2.0",
    "vosk-model-en-us-0.22": "Apache 2.0",
    "vosk-model-en-us-0.22-lgraph": "Apache 2.0",
    "vosk-model-en-us-0.42-gigaspeech": "Apache 2.0",
    "vosk-model-en-us-aspire-0.2": "Apache 2.0",
    "vosk-model-en-us-daanzu-20200905": "AGPL",
    "vosk-model-en-us-daanzu-20200905-lgraph": "AGPL",
    "vosk-model-en-us-librispeech-0.2": "Apache 2.0",
    "vosk-model-small-en-in-0.4": "Apache 2.0",
    "vosk-model-small-en-us-0.15": "Apache 2.0",
    "vosk-model-small-en-us-zamia-0.5": "LGPL-3.0",
    "vosk-model-small-eo-0.42": "Apache 2.0",
    "vosk-model-es-0.42": "Apache 2.0",
    "vosk-model-small-es-0.42": "Apache 2.0",
    "vosk-model-fa-0.42": "Apache 2.0",
    "vosk-model-fa-0.5": "Apache 2.0",
    "vosk-model-small-fa-0.42": "Apache 2.0",
    "vosk-model-small-fa-0.5": "Apache 2.0",
    "vosk-model-fr-0.22": "Apache 2.0",
    "vosk-model-fr-0.6-linto-2.2.0": "AGPL",
    "vosk-model-small-fr-0.22": "Apache 2.0",
    "vosk-model-small-fr-pguyot-0.3": "CC-BY-NC-SA 4.0",
    "vosk-model-gu-0.42": "Apache 2.0",
    "vosk-model-small-gu-0.42": "Apache 2.0",
    "vosk-model-hi-0.22": "Apache 2.0",
    "vosk-model-small-hi-0.22": "Apache 2.0",
    "vosk-model-it-0.22": "Apache 2.0",
    "vosk-model-small-it-0.22": "Apache 2.0",
    "vosk-model-ja-0.22": "Apache 2.0",
    "vosk-model-small-ja-0.22": "Apache 2.0",
    "vosk-model-ka-0.42": "Apache 2.0",
    "vosk-model-small-ka-0.42": "Apache 2.0",
    "vosk-model-small-ko-0.22": "Apache 2.0",
    "vosk-model-ky-0.42": "Apache 2.0",
    "vosk-model-small-ky-0.42": "Apache 2.0",
    "vosk-model-kz-0.42": "Apache 2.0",
    "vosk-model-small-kz-0.42": "Apache 2.0",
    "vosk-model-small-nl-0.22": "Apache 2.0",
    "vosk-model-nl-spraakherkenning-0.6": "CC-BY-NC-SA",
    "vosk-model-nl-spraakherkenning-0.6-lgraph": "CC-BY-NC-SA",
    "vosk-model-small-pl-0.22": "Apache 2.0",
    "vosk-model-pt-fb-v0.1.1-20220516_2113": "GPLv3.0",
    "vosk-model-small-pt-0.3": "Apache 2.0",
    "vosk-model-ru-0.10": "Apache 2.0",
    "vosk-model-ru-0.22": "Apache 2.0",
    "vosk-model-ru-0.42": "Apache 2.0",
    "vosk-model-small-ru-0.22": "Apache 2.0",
    "vosk-model-small-sv-rhasspy-0.15": "MIT",
    "vosk-model-small-te-0.42": "Apache 2.0",
    "vosk-model-small-tg-0.22": "Apache 2.0",
    "vosk-model-tg-0.22": "Apache 2.0",
    "vosk-model-tl-ph-generic-0.6": "CC-BY-NC-SA 4.0",
    "vosk-model-small-tr-0.3": "Apache 2.0",
    "vosk-model-uk-v3": "Apache 2.0",
    "vosk-model-uk-v3-lgraph": "Apache 2.0",
    "vosk-model-small-uk-v3-nano": "Apache 2.0",
    "vosk-model-small-uk-v3-small": "Apache 2.0",
    "vosk-model-small-uz-0.22": "Apache 2.0",
    "vosk-model-vn-0.4": "Apache 2.0",
    "vosk-model-small-vn-0.4": "Apache 2.0",
}


def supports_of(name: str) -> str:
    """What the model supports, from its name/kind."""
    notes = []
    if "-small-" in name:
        notes.append("compact")
    if "lgraph" in name:
        notes.append("small decoding graph")
    suffix = f" ({', '.join(notes)})" if notes else ""
    return f"Speech recognition{suffix}"


def get_ar_models():
    """ar (4 models)."""
    return [
        VoskModel(
            name="vosk-model-ar-0.22-linto-1.1.0",
            url="https://alphacephei.com/vosk/models/vosk-model-ar-0.22-linto-1.1.0.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-ar-mgb2-0.4",
            url="https://alphacephei.com/vosk/models/vosk-model-ar-mgb2-0.4.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-ar-tn-0.1-linto",
            url="https://alphacephei.com/vosk/models/vosk-model-ar-tn-0.1-linto.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-ar-tn-0.1-linto",
            url="https://alphacephei.com/vosk/models/vosk-model-small-ar-tn-0.1-linto.zip",
            kind="asr",
        ),
    ]


def get_br_models():
    """br (1 models)."""
    return [
        VoskModel(
            name="vosk-model-br-0.8",
            url="https://alphacephei.com/vosk/models/vosk-model-br-0.8.zip",
            kind="asr",
        ),
    ]


def get_ca_models():
    """ca (1 models)."""
    return [
        VoskModel(
            name="vosk-model-small-ca-0.4",
            url="https://alphacephei.com/vosk/models/vosk-model-small-ca-0.4.zip",
            kind="asr",
        ),
    ]


def get_cn_models():
    """cn (3 models)."""
    return [
        VoskModel(
            name="vosk-model-cn-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-cn-kaldi-multicn-0.15",
            url="https://alphacephei.com/vosk/models/vosk-model-cn-kaldi-multicn-0.15.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-cn-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip",
            kind="asr",
        ),
    ]


def get_cs_models():
    """cs (1 models)."""
    return [
        VoskModel(
            name="vosk-model-small-cs-0.4-rhasspy",
            url="https://alphacephei.com/vosk/models/vosk-model-small-cs-0.4-rhasspy.zip",
            kind="asr",
        ),
    ]


def get_de_models():
    """de (4 models)."""
    return [
        VoskModel(
            name="vosk-model-de-0.21",
            url="https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-de-tuda-0.6-900k",
            url="https://alphacephei.com/vosk/models/vosk-model-de-tuda-0.6-900k.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-de-0.15",
            url="https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-de-zamia-0.3",
            url="https://alphacephei.com/vosk/models/vosk-model-small-de-zamia-0.3.zip",
            kind="asr",
        ),
    ]


def get_el_models():
    """el (1 models)."""
    return [
        VoskModel(
            name="vosk-model-el-gr-0.7",
            url="https://alphacephei.com/vosk/models/vosk-model-el-gr-0.7.zip",
            kind="asr",
        ),
    ]


def get_en_models():
    """en (12 models)."""
    return [
        VoskModel(
            name="vosk-model-en-in-0.5",
            url="https://alphacephei.com/vosk/models/vosk-model-en-in-0.5.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-en-us-0.21",
            url="https://alphacephei.com/vosk/models/vosk-model-en-us-0.21.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-en-us-0.22-lgraph",
            url="https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-en-us-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-en-us-0.42-gigaspeech",
            url="https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-en-us-aspire-0.2",
            url="https://alphacephei.com/vosk/models/vosk-model-en-us-aspire-0.2.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-en-us-daanzu-20200905-lgraph",
            url="https://alphacephei.com/vosk/models/vosk-model-en-us-daanzu-20200905-lgraph.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-en-us-daanzu-20200905",
            url="https://alphacephei.com/vosk/models/vosk-model-en-us-daanzu-20200905.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-en-us-librispeech-0.2",
            url="https://alphacephei.com/vosk/models/vosk-model-en-us-librispeech-0.2.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-en-in-0.4",
            url="https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-en-us-0.15",
            url="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-en-us-zamia-0.5",
            url="https://alphacephei.com/vosk/models/vosk-model-small-en-us-zamia-0.5.zip",
            kind="asr",
        ),
    ]


def get_eo_models():
    """eo (1 models)."""
    return [
        VoskModel(
            name="vosk-model-small-eo-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-small-eo-0.42.zip",
            kind="asr",
        ),
    ]


def get_es_models():
    """es (2 models)."""
    return [
        VoskModel(
            name="vosk-model-es-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-es-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
            kind="asr",
        ),
    ]


def get_fa_models():
    """fa (4 models)."""
    return [
        VoskModel(
            name="vosk-model-fa-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-fa-0.42.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-fa-0.5",
            url="https://alphacephei.com/vosk/models/vosk-model-fa-0.5.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-fa-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-small-fa-0.42.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-fa-0.5",
            url="https://alphacephei.com/vosk/models/vosk-model-small-fa-0.5.zip",
            kind="asr",
        ),
    ]


def get_fr_models():
    """fr (4 models)."""
    return [
        VoskModel(
            name="vosk-model-fr-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-fr-0.6-linto-2.2.0",
            url="https://alphacephei.com/vosk/models/vosk-model-fr-0.6-linto-2.2.0.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-fr-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-fr-pguyot-0.3",
            url="https://alphacephei.com/vosk/models/vosk-model-small-fr-pguyot-0.3.zip",
            kind="asr",
        ),
    ]


def get_gu_models():
    """gu (2 models)."""
    return [
        VoskModel(
            name="vosk-model-gu-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-gu-0.42.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-gu-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-small-gu-0.42.zip",
            kind="asr",
        ),
    ]


def get_hi_models():
    """hi (2 models)."""
    return [
        VoskModel(
            name="vosk-model-hi-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-hi-0.22.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-hi-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
            kind="asr",
        ),
    ]


def get_it_models():
    """it (2 models)."""
    return [
        VoskModel(
            name="vosk-model-it-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-it-0.22.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-it-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip",
            kind="asr",
        ),
    ]


def get_ja_models():
    """ja (2 models)."""
    return [
        VoskModel(
            name="vosk-model-ja-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-ja-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip",
            kind="asr",
        ),
    ]


def get_ka_models():
    """ka (2 models)."""
    return [
        VoskModel(
            name="vosk-model-ka-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-ka-0.42.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-ka-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-small-ka-0.42.zip",
            kind="asr",
        ),
    ]


def get_ko_models():
    """ko (1 models)."""
    return [
        VoskModel(
            name="vosk-model-small-ko-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip",
            kind="asr",
        ),
    ]


def get_ky_models():
    """ky (2 models)."""
    return [
        VoskModel(
            name="vosk-model-ky-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-ky-0.42.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-ky-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-small-ky-0.42.zip",
            kind="asr",
        ),
    ]


def get_kz_models():
    """kz (4 models; incl. 2 unlisted, from the Derur mirror)."""
    return [
        VoskModel(
            name="vosk-model-kz-0.15",
            url="https://alphacephei.com/vosk/models/vosk-model-kz-0.15.zip",
            kind="asr",  # not listed on the models page; kept at the old
            # URL (found in the https://huggingface.co/Derur/vosk-models
            # mirror); included as part of the union of both sources
        ),
        VoskModel(
            name="vosk-model-small-kz-0.15",
            url="https://alphacephei.com/vosk/models/vosk-model-small-kz-0.15.zip",
            kind="asr",  # not listed on the models page; kept at the old
            # URL (found in the https://huggingface.co/Derur/vosk-models
            # mirror); included as part of the union of both sources
        ),
        VoskModel(
            name="vosk-model-kz-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-kz-0.42.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-kz-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-small-kz-0.42.zip",
            kind="asr",
        ),
    ]


def get_nl_models():
    """nl (3 models)."""
    return [
        VoskModel(
            name="vosk-model-nl-spraakherkenning-0.6-lgraph",
            url="https://alphacephei.com/vosk/models/vosk-model-nl-spraakherkenning-0.6-lgraph.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-nl-spraakherkenning-0.6",
            url="https://alphacephei.com/vosk/models/vosk-model-nl-spraakherkenning-0.6.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-nl-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-small-nl-0.22.zip",
            kind="asr",
        ),
    ]


def get_pl_models():
    """pl (1 models)."""
    return [
        VoskModel(
            name="vosk-model-small-pl-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-small-pl-0.22.zip",
            kind="asr",
        ),
    ]


def get_pt_models():
    """pt (2 models)."""
    return [
        VoskModel(
            name="vosk-model-pt-fb-v0.1.1-20220516_2113",
            url="https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-20220516_2113.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-pt-0.3",
            url="https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
            kind="asr",
        ),
    ]


def get_ru_models():
    """ru (4 models)."""
    return [
        VoskModel(
            name="vosk-model-ru-0.10",
            url="https://alphacephei.com/vosk/models/vosk-model-ru-0.10.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-ru-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-ru-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-ru-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
            kind="asr",
        ),
    ]


def get_sv_models():
    """sv (1 models)."""
    return [
        VoskModel(
            name="vosk-model-small-sv-rhasspy-0.15",
            url="https://alphacephei.com/vosk/models/vosk-model-small-sv-rhasspy-0.15.zip",
            kind="asr",
        ),
    ]


def get_te_models():
    """te (1 models)."""
    return [
        VoskModel(
            name="vosk-model-small-te-0.42",
            url="https://alphacephei.com/vosk/models/vosk-model-small-te-0.42.zip",
            kind="asr",
        ),
    ]


def get_tg_models():
    """tg (2 models)."""
    return [
        VoskModel(
            name="vosk-model-small-tg-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-small-tg-0.22.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-tg-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-tg-0.22.zip",
            kind="asr",
        ),
    ]


def get_tl_models():
    """tl (1 models)."""
    return [
        VoskModel(
            name="vosk-model-tl-ph-generic-0.6",
            url="https://alphacephei.com/vosk/models/vosk-model-tl-ph-generic-0.6.zip",
            kind="asr",
        ),
    ]


def get_tr_models():
    """tr (1 models)."""
    return [
        VoskModel(
            name="vosk-model-small-tr-0.3",
            url="https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip",
            kind="asr",
        ),
    ]


def get_uk_models():
    """uk (4 models)."""
    return [
        VoskModel(
            name="vosk-model-small-uk-v3-nano",
            url="https://alphacephei.com/vosk/models/vosk-model-small-uk-v3-nano.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-small-uk-v3-small",
            url="https://alphacephei.com/vosk/models/vosk-model-small-uk-v3-small.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-uk-v3-lgraph",
            url="https://alphacephei.com/vosk/models/vosk-model-uk-v3-lgraph.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-uk-v3",
            url="https://alphacephei.com/vosk/models/vosk-model-uk-v3.zip",
            kind="asr",
        ),
    ]


def get_uz_models():
    """uz (1 models)."""
    return [
        VoskModel(
            name="vosk-model-small-uz-0.22",
            url="https://alphacephei.com/vosk/models/vosk-model-small-uz-0.22.zip",
            kind="asr",
        ),
    ]


def get_vn_models():
    """vn (2 models)."""
    return [
        VoskModel(
            name="vosk-model-small-vn-0.4",
            url="https://alphacephei.com/vosk/models/vosk-model-small-vn-0.4.zip",
            kind="asr",
        ),
        VoskModel(
            name="vosk-model-vn-0.4",
            url="https://alphacephei.com/vosk/models/vosk-model-vn-0.4.zip",
            kind="asr",
        ),
    ]


def get_all_models():
    ans = []
    ans += get_ar_models()
    ans += get_br_models()
    ans += get_ca_models()
    ans += get_cn_models()
    ans += get_cs_models()
    ans += get_de_models()
    ans += get_el_models()
    ans += get_en_models()
    ans += get_eo_models()
    ans += get_es_models()
    ans += get_fa_models()
    ans += get_fr_models()
    ans += get_gu_models()
    ans += get_hi_models()
    ans += get_it_models()
    ans += get_ja_models()
    ans += get_ka_models()
    ans += get_ko_models()
    ans += get_ky_models()
    ans += get_kz_models()
    ans += get_nl_models()
    ans += get_pl_models()
    ans += get_pt_models()
    ans += get_ru_models()
    ans += get_sv_models()
    ans += get_te_models()
    ans += get_tg_models()
    ans += get_tl_models()
    ans += get_tr_models()
    ans += get_uk_models()
    ans += get_uz_models()
    ans += get_vn_models()

    for i, m in enumerate(ans):
        m.index = i
        m.lang = lang_of(m.name)  # asr/<lang>/<name>.zip on huggingface
        m.license = LICENSES.get(m.name, "")
        m.wav, m.expect = TEST_WAVS.get(m.lang, ("", ""))

    return ans
