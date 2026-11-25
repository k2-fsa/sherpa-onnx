#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# Generate ./generate.sh: download every vosk model listed on
# https://alphacephei.com/vosk/models and upload it to
# https://huggingface.co/csukuangfj/vosk-models -- each model is uploaded
# as soon as it is downloaded (see generate.sh.in).
#
# Only the ASR models are mirrored (no speaker-id / text models).
# The list below mirrors the "Vosk Speech Recognition Models" page
# (that page is the authoritative source; https://huggingface.co/Derur/vosk-models
# is only an incomplete copy).  To refresh the list:
#
#   curl -sL https://alphacephei.com/vosk/models \
#     | grep -oE 'href="[^"]*\.zip"' | sed 's/href="//;s/"//' | sort -u
#
# Sharding follows scripts/piper/generate.py: pass --total/--index to
# split the model list across CI runners.

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import jinja2

HF_REPO = "csukuangfj/vosk-models"


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
    parser.add_argument(
        "--info",
        action="store_true",
        help="Write release/asr/<lang>/<name>.md info pages for every model "
        "with a local ls -lhR listing (from --listing-dir); generate.sh "
        "uploads each to asr/<lang>/<name>.md on csukuangfj/vosk-models",
    )
    parser.add_argument(
        "--only",
        default="",
        help="With --info: render the info page of just this model "
        "(generate.sh does this per model, right before uploading it)",
    )
    parser.add_argument(
        "--readme",
        action="store_true",
        help="Assemble README.md (tables linking to the info/ pages), "
        "instead of rendering generate.sh",
    )
    parser.add_argument(
        "--listing-dir",
        default="listing",
        help="Directory holding <model>.txt ls -lhR listings",
    )
    parser.add_argument(
        "--upload",
        nargs="+",
        default=[],
        metavar="LOCAL:REMOTE",
        help="Upload the given files to " + HF_REPO + " in ONE commit "
        "(the repo allows only 128 commits per hour, so per-file commits "
        "do not scale to 78 models x 4 files), then exit",
    )
    parser.add_argument(
        "--message",
        default="upload",
        help="Commit message for --upload",
    )
    return parser.parse_args()


def hf_commit(pairs, message):
    """Upload all <local>:<remote> pairs to HF_REPO in ONE commit."""
    import time

    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:  # huggingface_hub < 0.23
        from huggingface_hub.utils import HfHubHTTPError
    from huggingface_hub import CommitOperationAdd, HfApi

    operations = []
    for pair in pairs:
        local, remote = pair.split(":", 1)
        operations.append(
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local)
        )
    api = HfApi()
    # HF allows only 128 commit creations per hour per repo; when that is
    # used up it answers 429 "retry in about 1 hour" -- so wait it out
    # (minutes-scale bash retries cannot bridge an hour-long window)
    for attempt in range(12):
        try:
            commit_info = api.create_commit(
                repo_id=HF_REPO,
                operations=operations,
                commit_message=message,
                token=os.environ["HF_TOKEN"],
            )
            print(f"{commit_info}: {len(operations)} files in one commit")
            return
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if (status != 429 and "429" not in str(e)) or attempt == 11:
                raise
            print(
                f"429 from huggingface (commit rate limit); "
                f"waiting 300s (attempt {attempt + 1}/12)",
                flush=True,
            )
            time.sleep(300)


@dataclass
class VoskModel:
    name: str  # e.g. vosk-model-small-cn-0.22 (the dir inside the zip)
    url: str  # https://alphacephei.com/vosk/models/<name>.zip
    kind: str = "asr"
    lang: str = ""  # language code, filled by get_all_models()
    index: int = 0


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
    if parts[1] == "recasepunc":
        return parts[2]
    if parts[1] == "model" and parts[2] == "spk":
        return "spk"
    i = 2
    if parts[i] == "small":
        i += 1
    return parts[i]


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

    return ans


CONTAINS = {
    "am": "acoustic model (`am/`)",
    "conf": "configs (`conf/`)",
    "graph": "decoding graph (`graph/`)",
    "ivector": "i-vector extractor (`ivector/`)",
    "rescore": "rescoring extras (`rescore/`)",
    "rnnlm": "RNNLM (`rnnlm/`)",
    "README": "model notes (`README`)",
}


def contains_of(listing: str) -> str:
    """What the model folder contains, from its ls -lhR listing (the
    first block lists the folder's top-level entries)."""
    entries = []
    lines = listing.splitlines()
    i = 1 if lines and lines[0].rstrip().endswith(":") else 0
    for line in lines[i:]:
        if line.rstrip().endswith(":"):
            break  # next folder block: the top-level listing is done
        if not line.strip() or line.startswith("total"):
            continue
        entries.append(line.split()[-1])
    parts = [desc for name, desc in CONTAINS.items() if name in entries]
    return "; ".join(parts) if parts else "(see listing)"


def render_info(all_model_list, listing_dir: Path, only: str = ""):
    """release/asr/<lang>/<name>.md for every model with a local listing:
    what the zip contains and its ls -lhR file sizes.  With `only`, just
    that one model (generate.sh renders + uploads each model as it goes)."""
    listing_dir = Path(listing_dir)
    out = Path("release")  # info pages land next to their zip:
    # release/asr/<lang>/<name>.md -> uploaded as asr/<lang>/<name>.md
    env = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )
    with open("./info.md.in") as f:
        template = env.from_string(f.read())
    made = 0
    for m in all_model_list:
        if only and m.name != only:
            continue
        p = listing_dir / f"{m.name}.txt"
        if not p.is_file():
            continue
        page = out / "asr" / lang_of(m.name) / f"{m.name}.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        tree_p = listing_dir / f"{m.name}.tree.txt"
        page.write_text(
            template.render(
                name=m.name,
                url=m.url,
                supports=supports_of(m.name),
                contains=contains_of(p.read_text()),
                listing=p.read_text().rstrip(),
                tree=tree_p.read_text().rstrip()
                if tree_p.is_file()
                else "(tree listing not available)",
            )
        )
        made += 1
    print(f"info/: {made} model pages written")


def render_readme(all_model_list):
    """README.md: table of contents and one table per language, each row
    linking to the model's info/ page (which lists the files in its zip)."""
    by_lang = {}
    for m in all_model_list:
        by_lang.setdefault(lang_of(m.name), []).append(m)
    languages = []
    for lang in sorted(by_lang):
        title = "{} (`{}`)".format(LANG_NAMES.get(lang, lang), lang)
        languages.append(
            {
                "title": title,
                "anchor": re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-"),
                "models": [
                    {
                        "name": m.name,
                        "url": m.url,
                        "lang": lang_of(m.name),
                        "supports": supports_of(m.name),
                    }
                    for m in by_lang[lang]
                ],
            }
        )
    environment = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )
    with open("./README.md.in") as f:
        template = environment.from_string(f.read())
    with open("./README.md", "w") as f:
        f.write(template.render(languages=languages))
    print(f"README.md written for {len(all_model_list)} models")


def main():
    args = get_args()
    if args.upload:
        hf_commit(args.upload, args.message)
        return
    index = args.index
    total = args.total
    assert 0 <= index < total, (index, total)

    all_model_list = get_all_models()

    # Note: --info/--readme must return BEFORE generate.sh is rendered:
    # generate.sh calls `generate.py --info --only <name>` while itself
    # running, and bash reads its script file incrementally -- rewriting
    # it mid-run would corrupt execution.
    if args.info:
        render_info(all_model_list, Path(args.listing_dir), only=args.only)
        return
    if args.readme:
        render_readme(all_model_list)
        return

    num_models = len(all_model_list)
    num_per_runner = num_models // total
    if num_per_runner <= 0:
        raise ValueError(f"num_models: {num_models}, num_runners: {total}")

    start = index * num_per_runner
    end = start + num_per_runner

    remaining = num_models - total * num_per_runner

    print(f"{index}/{total}: {start}-{end}/{num_models}")

    d = dict()
    d["model_list"] = all_model_list[start:end]

    if index < remaining:
        s = total * num_per_runner + index
        d["model_list"].append(all_model_list[s])
        print(f"{s}/{num_models}")

    filename_list = [
        "./generate.sh",
    ]
    for filename in filename_list:
        environment = jinja2.Environment()
        if not Path(f"{filename}.in").is_file():
            print(f"skip {filename}")
            continue

        with open(f"{filename}.in") as f:
            s = f.read()
        template = environment.from_string(s)

        s = template.render(**d)
        with open(filename, "w") as f:
            print(s, file=f)

    print(f"There are {len(all_model_list)} models")
    for m in d["model_list"]:
        print(m.index, m.kind, m.name)


if __name__ == "__main__":
    main()
