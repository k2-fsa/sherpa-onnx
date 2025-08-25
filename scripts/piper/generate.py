#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
from dataclasses import dataclass
from pathlib import Path

import jinja2

"""
TODO:
 - add https://huggingface.co/csukuangfj/vits-piper-en_US-glados
"""


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
class PiperModel:
    # For en_GB-semaine-medium
    name: str  # semaine
    kind: str  # e.g. medium
    sr: int  # sample rate
    ns: int  # number of speakers
    lang: str = ""  # e.g., en_GB
    cmd: str = ""
    model_name: str = ""
    text: str = ""
    index: int = 0
    url: str = ""


# arabic
def get_ar_models():
    ar_jo = [
        PiperModel(name="kareem", kind="low", sr=16000, ns=1),
        PiperModel(name="kareem", kind="medium", sr=22050, ns=1),
    ]

    for m in ar_jo:
        m.lang = "ar_JO"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = ar_jo

    for m in ans:
        m.text = "كيف حالك اليوم؟"
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# catlan
def get_ca_models():
    ca_es = [
        PiperModel(name="upc_ona", kind="medium", sr=22050, ns=1),
        PiperModel(name="upc_ona", kind="x_low", sr=16000, ns=1),
        PiperModel(name="upc_pau", kind="x_low", sr=16000, ns=1),
    ]

    for m in ca_es:
        m.lang = "ca_ES"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = ca_es

    for m in ans:
        m.text = "Si vols estar ben servit, fes-te tu mateix el llit"
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# czech
def get_cs_models():
    cs_cz = [
        PiperModel(name="jirka", kind="low", sr=16000, ns=1),
        PiperModel(name="jirka", kind="medium", sr=22050, ns=1),
    ]

    for m in cs_cz:
        m.lang = "cs_CZ"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = cs_cz

    for m in ans:
        m.text = "Co můžeš udělat dnes, neodkládej na zítřek. "
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# welsh
def get_cy_models():
    cy_gb = [
        PiperModel(name="bu_tts", kind="medium", sr=22050, ns=7),
        PiperModel(name="gwryw_gogleddol", kind="medium", sr=22050, ns=1),
    ]

    for m in cy_gb:
        m.lang = "cy_GB"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = cy_gb

    for m in ans:
        m.text = "Ni all y gwynt ei hunan ei ddilyn, ac felly mae’n rhaid i’r gŵyr ddod i’r gorwel i weld y llwybr yn gyfarwydd"
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# danish
def get_da_models():
    da_dk = [
        PiperModel(name="talesyntese", kind="medium", sr=22050, ns=1),
    ]

    for m in da_dk:
        m.lang = "da_DK"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = da_dk

    for m in ans:
        m.text = (
            "Hvis du går langsomt, men aldrig stopper, når du ender frem til dit mål."
        )
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# greek
def get_el_models():
    el_gr = [
        PiperModel(name="rapunzelina", kind="low", sr=16000, ns=1),
    ]

    for m in el_gr:
        m.lang = "el_GR"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = el_gr

    for m in ans:
        m.text = (
            "Όταν το δέντρο είναι μικρό, το στρέβλεις· όταν είναι μεγάλο, το λυγίζεις."
        )
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# spanish
def get_es_models():
    es_ES = [
        PiperModel(name="carlfm", kind="x_low", sr=16000, ns=1),
        PiperModel(name="davefx", kind="medium", sr=22050, ns=1),
        PiperModel(name="sharvard", kind="medium", sr=22050, ns=2),
    ]

    es_ES.extend(
        [
            # https://github.com/rhasspy/piper/issues/187#issuecomment-1802216304
            # https://drive.google.com/file/d/12tNCCyd0Hf5jsyqCw8828kLSHHx5LOw9/view
            PiperModel(
                name="glados",
                kind="medium",
                sr=22050,
                ns=1,
                cmd="""
                   wget -qq https://huggingface.co/csukuangfj/vits-piper-es_ES-glados-medium/resolve/main/es_ES-glados-medium.onnx
                   wget -qq https://huggingface.co/csukuangfj/vits-piper-es_ES-glados-medium/resolve/main/es_ES-glados-medium.onnx.json
                   wget -qq https://huggingface.co/csukuangfj/vits-piper-es_ES-glados-medium/resolve/main/README.md
                   """,
                url="https://github.com/rhasspy/piper/issues/187#issuecomment-1802216304",
            ),
        ]
    )

    es_ES.extend(
        [
            PiperModel(
                name="miro",
                kind="high",
                sr=22050,
                ns=1,
                cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_es-ES_miro/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_es-ES_miro" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md
                   echo "\n\n#License\n\n" >> README

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md


                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_es-ES_miro/resolve/main/miro_es-ES.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_es-ES_miro/resolve/main/miro_es-ES.onnx.json

                   mv miro_es-ES.onnx es_ES-miro-high.onnx
                   mv miro_es-ES.onnx.json es_ES-miro-high.onnx.json
                   """,
                url="https://huggingface.co/OpenVoiceOS/pipertts_es-ES_miro",
            ),
        ]
    )

    es_MX = [
        PiperModel(name="ald", kind="medium", sr=22050, ns=1),
        PiperModel(name="claude", kind="high", sr=22050, ns=1),
    ]

    # Argentina
    es_AR = [
        PiperModel(name="daniela", kind="high", sr=22050, ns=1),
    ]

    for m in es_ES:
        m.lang = "es_ES"

    for m in es_MX:
        m.lang = "es_MX"

    for m in es_AR:
        m.lang = "es_AR"

    ans = es_ES + es_MX + es_AR

    for m in ans:
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        m.text = "Cuando te encuentres ante una puerta cerrada, no olvides que a veces el destino cierra una puerta para que te desvíes hacia un camino que lleva a una ventana que nunca habrías encontrado por tu cuenta."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# persian
def get_fa_models():
    fa_IR = [
        PiperModel(name="amir", kind="medium", sr=22050, ns=1),
        PiperModel(name="ganji", kind="medium", sr=22050, ns=1),
        PiperModel(name="ganji_adabi", kind="medium", sr=22050, ns=1),
        PiperModel(name="gyro", kind="medium", sr=22050, ns=1),
        PiperModel(name="reza_ibrahim", kind="medium", sr=22050, ns=1),
    ]

    for m in fa_IR:
        m.lang = "fa_IR"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = fa_IR

    for m in ans:
        m.text = "همانطور که کوه ها در برابر باد و باران پایدارند، اما به مرور زمان خرد و پخش می شوند، انسان نیز باید در برابر مشکلات قوی باشد، اما با خرد و خویشتن داری در زندگی به پیش برود."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# finnish
def get_fi_models():
    fi_FI = [
        PiperModel(name="harri", kind="low", sr=16000, ns=1),
        PiperModel(name="harri", kind="medium", sr=22050, ns=1),
    ]

    for m in fi_FI:
        m.lang = "fi_FI"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = fi_FI

    for m in ans:
        m.text = "Sateenkaaren päässä on kultaa, mutta vain ne, jotka siihen uskovat, voivat sen löytää."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# french
def get_fr_models():
    fr_FR = [
        PiperModel(name="gilles", kind="low", sr=16000, ns=1),
        PiperModel(name="siwis", kind="low", sr=16000, ns=1),
        PiperModel(name="siwis", kind="medium", sr=22050, ns=1),
        PiperModel(name="tom", kind="medium", sr=44100, ns=1),
        PiperModel(name="upmc", kind="medium", sr=22050, ns=2),
    ]

    fr_FR.extend(
        [
            PiperModel(
                name="tjiho",
                kind=f"model{k}",
                sr=44100,
                ns=1,
                cmd=f"""
                   wget -qq https://huggingface.co/csukuangfj/vits-piper-fr_FR-tjiho-model{k}/resolve/main/fr_FR-tjiho-model{k}.onnx
                   wget -qq https://huggingface.co/csukuangfj/vits-piper-fr_FR-tjiho-model{k}/resolve/main/fr_FR-tjiho-model{k}.onnx.json
                   wget -qq https://huggingface.co/csukuangfj/vits-piper-fr_FR-tjiho-model{k}/resolve/main/LICENSE.txt
                   wget -qq https://huggingface.co/csukuangfj/vits-piper-fr_FR-tjiho-model{k}/resolve/main/MODEL_CARD
                   """,
                url=f"https://huggingface.co/csukuangfj/vits-piper-fr_FR-tjiho-model{k}/tree/main",
            )
            for k in [1, 2, 3]
        ]
    )

    fr_FR += [
        PiperModel(
            name="miro",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_fr-FR_miro/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_fr-FR_miro" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md



                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_fr-FR_miro/resolve/main/miro_fr-FR.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_fr-FR_miro/resolve/main/miro_fr-FR.onnx.json

                   mv miro_fr-FR.onnx fr_FR-miro-high.onnx
                   mv miro_fr-FR.onnx.json fr_FR-miro-high.onnx.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_fr-FR_miro",
        ),
    ]

    for m in fr_FR:
        m.lang = "fr_FR"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = fr_FR

    for m in ans:
        m.text = "Pas de nouvelles, bonnes nouvelles."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# hindi
def get_hi_models():
    hi_IN = [
        PiperModel(name="pratham", kind="medium", sr=22050, ns=1),
        PiperModel(name="priyamvada", kind="medium", sr=22050, ns=1),
        PiperModel(name="rohan", kind="medium", sr=22050, ns=1),
    ]

    for m in hi_IN:
        m.lang = "hi_IN"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = hi_IN

    for m in ans:
        m.text = "यह मत पूछो कि तुम्हारा देश तुम्हारे लिए क्या कर सकता है। यह पूछो कि तुम अपने देश के लिए क्या कर सकते हो।"
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# hungarian
def get_hu_models():
    hu_HU = [
        PiperModel(name="anna", kind="medium", sr=22050, ns=1),
        PiperModel(name="berta", kind="medium", sr=22050, ns=1),
        PiperModel(name="imre", kind="medium", sr=22050, ns=1),
    ]

    for m in hu_HU:
        m.lang = "hu_HU"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = hu_HU

    for m in ans:
        m.text = "Ha északról fúj a szél, a lányok nem lógnak együtt."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# icelandic
def get_is_models():
    is_IS = [
        PiperModel(name="bui", kind="medium", sr=22050, ns=1),
        PiperModel(name="salka", kind="medium", sr=22050, ns=1),
        PiperModel(name="steinn", kind="medium", sr=22050, ns=1),
        PiperModel(name="ugla", kind="medium", sr=22050, ns=1),
    ]

    for m in is_IS:
        m.lang = "is_IS"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = is_IS

    for m in ans:
        m.text = "Farðu með allt, eða farðu ekki."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# italian
def get_it_models():
    it_IT = [
        PiperModel(name="paola", kind="medium", sr=22050, ns=1),
        PiperModel(name="riccardo", kind="x_low", sr=16000, ns=1),
    ]

    it_IT += [
        PiperModel(
            name="miro",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_it-IT_miro/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_it-IT_miro" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md



                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_it-IT_miro/resolve/main/miro_it-IT.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_it-IT_miro/resolve/main/miro_it-IT.onnx.json

                   mv miro_it-IT.onnx it_IT-miro-high.onnx
                   mv miro_it-IT.onnx.json it_IT-miro-high.onnx.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_it-IT_miro",
        ),
        PiperModel(
            name="dii",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_it-IT_dii/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_it-IT_dii" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md



                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_it-IT_dii/resolve/main/dii_it-IT.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_it-IT_dii/resolve/main/dii_it-IT.onnx.json

                   mv dii_it-IT.onnx it_IT-dii-high.onnx
                   mv dii_it-IT.onnx.json it_IT-dii-high.onnx.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_it-IT_dii",
        ),
    ]

    for m in it_IT:
        m.lang = "it_IT"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = it_IT

    for m in ans:
        m.text = (
            "Se vuoi andare veloce, vai da solo; se vuoi andare lontano, vai insieme."
        )
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# georgian
def get_ka_models():
    ka_GE = [
        PiperModel(name="natia", kind="medium", sr=22050, ns=1),
    ]

    for m in ka_GE:
        m.lang = "ka_GE"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = ka_GE

    for m in ans:
        m.text = "ღვინო თბილისში, საქართველო სამტრედში"
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# kazakh
def get_kk_models():
    kk_KZ = [
        PiperModel(name="iseke", kind="x_low", sr=16000, ns=1),
        PiperModel(name="issai", kind="high", sr=22050, ns=6),
        PiperModel(name="raya", kind="x_low", sr=16000, ns=1),
    ]

    for m in kk_KZ:
        m.lang = "kk_KZ"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = kk_KZ

    for m in ans:
        m.text = "Әлемнің жұлдыздары сенің көзің, жаным."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# luxembourgish
def get_lb_models():
    lb_LU = [
        PiperModel(name="marylux", kind="medium", sr=22050, ns=1),
    ]

    for m in lb_LU:
        m.lang = "lb_LU"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = lb_LU

    for m in ans:
        m.text = "Op der Haaptstrooss sinn all Stroossen Brécken, awer d'Dier kann iwwerall erreecht ginn."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# latvian
def get_lv_models():
    lv_LV = [
        PiperModel(name="aivars", kind="medium", sr=22050, ns=1),
    ]

    for m in lv_LV:
        m.lang = "lv_LV"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = lv_LV

    for m in ans:
        m.text = "Zeme nenes augļus, ja tēvs sēj, bet māte auž."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# malayalam
def get_ml_models():
    ml_IN = [
        PiperModel(name="arjun", kind="medium", sr=22050, ns=1),
        PiperModel(name="meera", kind="medium", sr=22050, ns=1),
    ]

    for m in ml_IN:
        m.lang = "ml_IN"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = ml_IN

    for m in ans:
        m.text = "മണ്ണ് മരിക്കുമ്പോൾ കാട്ടിലെ വെള്ളവും മരിക്കുന്നു."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# Nepali
def get_ne_models():
    ne_NP = [
        PiperModel(name="chitwan", kind="medium", sr=22050, ns=1),
        PiperModel(name="google", kind="medium", sr=22050, ns=18),
        PiperModel(name="google", kind="x_low", sr=16000, ns=18),
    ]

    for m in ne_NP:
        m.lang = "ne_NP"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = ne_NP

    for m in ans:
        m.text = "घाँसको पातले पहाडलाई अभिवादन गर्दै झुक्छ।"
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# dutch
def get_nl_models():
    nl_BE = [
        PiperModel(name="nathalie", kind="medium", sr=22050, ns=1),
        PiperModel(name="nathalie", kind="x_low", sr=16000, ns=1),
    ]

    nl_NL = [
        PiperModel(name="pim", kind="medium", sr=22050, ns=1),
        PiperModel(name="ronnie", kind="medium", sr=22050, ns=1),
    ]

    nl_NL += [
        PiperModel(
            name="miro",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_nl-NL_miro/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_nl-NL_miro" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md



                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_nl-NL_miro/resolve/main/miro_nl-NL.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_nl-NL_miro/resolve/main/miro_nl-NL.onnx.json

                   mv miro_nl-NL.onnx nl_NL-miro-high.onnx
                   mv miro_nl-NL.onnx.json nl_NL-miro-high.onnx.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_nl-NL_miro",
        ),
        PiperModel(
            name="dii",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_nl-NL_dii/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_nl-NL_dii" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md



                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_nl-NL_dii/resolve/main/dii_nl-NL.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_nl-NL_dii/resolve/main/dii_nl-NL.onnx.json

                   mv dii_nl-NL.onnx nl_NL-dii-high.onnx
                   mv dii_nl-NL.onnx.json nl_NL-dii-high.onnx.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_nl-NL_dii",
        ),
    ]

    for m in nl_BE:
        m.lang = "nl_BE"

    for m in nl_NL:
        m.lang = "nl_NL"

    ans = nl_BE + nl_NL

    for m in ans:
        m.text = "God schiep het water, maar de Nederlander schiep de dijk"

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# norwegian
def get_no_models():
    no_NO = [
        PiperModel(name="talesyntese", kind="medium", sr=22050, ns=1),
    ]

    for m in no_NO:
        m.lang = "no_NO"

    ans = no_NO

    for m in ans:
        m.text = "Uskyldig kan stormen veroorzaken"

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# polish
def get_pl_models():
    pl_PL = [
        PiperModel(name="darkman", kind="medium", sr=22050, ns=1),
        PiperModel(name="gosia", kind="medium", sr=22050, ns=1),
        PiperModel(name="mc_speech", kind="medium", sr=22050, ns=1),
    ]

    pl_PL.extend(
        [
            PiperModel(
                name="jarvis_wg_glos",
                kind="medium",
                sr=22050,
                ns=1,
                cmd="""
                   wget -qq https://huggingface.co/WitoldG/polish_piper_models/resolve/main/pl_PL-jarvis_wg_glos-medium.onnx
                   wget -qq https://huggingface.co/WitoldG/polish_piper_models/resolve/main/pl_PL-jarvis_wg_glos-medium.onnx.json
                   wget -qq https://huggingface.co/WitoldG/polish_piper_models/resolve/main/README.md
                   """,
                url="https://github.com/k2-fsa/sherpa-onnx/issues/2402",
            ),
            PiperModel(
                name="justyna_wg_glos",
                kind="medium",
                sr=22050,
                ns=1,
                cmd="""
                   wget -qq https://huggingface.co/WitoldG/polish_piper_models/resolve/main/pl_PL-justyna_wg_glos-medium.onnx
                   wget -qq https://huggingface.co/WitoldG/polish_piper_models/resolve/main/pl_PL-justyna_wg_glos-medium.onnx.json
                   wget -qq https://huggingface.co/WitoldG/polish_piper_models/resolve/main/README.md
                   """,
                url="https://github.com/k2-fsa/sherpa-onnx/issues/2402",
            ),
            PiperModel(
                name="meski_wg_glos",
                kind="medium",
                sr=22050,
                ns=1,
                cmd="""
                   wget -qq https://huggingface.co/WitoldG/polish_piper_models/resolve/main/pl_PL-meski_wg_glos-medium.onnx
                   wget -qq https://huggingface.co/WitoldG/polish_piper_models/resolve/main/pl_PL-meski_wg_glos-medium.onnx.json
                   wget -qq https://huggingface.co/WitoldG/polish_piper_models/resolve/main/README.md
                   """,
                url="https://github.com/k2-fsa/sherpa-onnx/issues/2402",
            ),
            PiperModel(
                name="zenski_wg_glos",
                kind="medium",
                sr=22050,
                ns=1,
                cmd="""
                   wget -qq https://huggingface.co/WitoldG/polish_piper_models/resolve/main/pl_PL-zenski_wg_glos-medium.onnx
                   wget -qq https://huggingface.co/WitoldG/polish_piper_models/resolve/main/pl_PL-zenski_wg_glos-medium.onnx.json
                   wget -qq https://huggingface.co/WitoldG/polish_piper_models/resolve/main/README.md
                   """,
                url="https://github.com/k2-fsa/sherpa-onnx/issues/2402",
            ),
        ]
    )

    for m in pl_PL:
        m.lang = "pl_PL"

    ans = pl_PL

    for m in ans:
        m.text = "Nieważne, za kogo walczysz, i tak popełnisz błąd"

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# Portuguese
def get_pt_models():
    pt_BR = [
        PiperModel(name="cadu", kind="medium", sr=22050, ns=1),
        PiperModel(name="edresson", kind="low", sr=16000, ns=1),
        PiperModel(name="faber", kind="medium", sr=22050, ns=1),
        PiperModel(name="jeff", kind="medium", sr=22050, ns=1),
    ]

    pt_PT = [
        PiperModel(
            name="tugao",
            kind="medium",
            sr=22050,
            ns=1,
            cmd="""
                    wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_PT/tugão/medium/pt_PT-tugão-medium.onnx
                    wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_PT/tugão/medium/pt_PT-tugão-medium.onnx.json
                    wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_PT/tugão/medium/MODEL_CARD

                    mv pt_PT-tugão-medium.onnx pt_PT-tugao-medium.onnx
                    mv pt_PT-tugão-medium.onnx.json pt_PT-tugao-medium.onnx.json
                   """,
            url="https://huggingface.co/rhasspy/piper-voices/tree/main/pt/pt_PT/tugão/medium",
        ),
    ]

    pt_PT += [
        PiperModel(
            name="miro",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_miro/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_miro" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md



                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_miro/resolve/main/miro_pt-PT.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_miro/resolve/main/miro_pt-PT.onnx.json

                   mv miro_pt-PT.onnx pt_PT-miro-high.onnx
                   mv miro_pt-PT.onnx.json pt_PT-miro-high.onnx.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_miro",
        ),
        PiperModel(
            name="dii",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_dii/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_dii" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md



                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_dii/resolve/main/dii_pt-PT.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_dii/resolve/main/dii_pt-PT.onnx.json

                   mv dii_pt-PT.onnx pt_PT-dii-high.onnx
                   mv dii_pt-PT.onnx.json pt_PT-dii-high.onnx.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_dii",
        ),
    ]

    pt_BR += [
        PiperModel(
            name="miro",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_miro/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_miro" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md



                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_miro/resolve/main/miro_pt-BR.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_miro/resolve/main/miro_pt-BR.onnx.json

                   mv miro_pt-BR.onnx pt_BR-miro-high.onnx
                   mv miro_pt-BR.onnx.json pt_BR-miro-high.onnx.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_miro",
        ),
        PiperModel(
            name="dii",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_dii/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_dii" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md



                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_dii/resolve/main/dii_pt-BR.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_dii/resolve/main/dii_pt-BR.onnx.json

                   mv dii_pt-BR.onnx pt_BR-dii-high.onnx
                   mv dii_pt-BR.onnx.json pt_BR-dii-high.onnx.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_dii",
        ),
    ]

    for m in pt_BR:
        m.lang = "pt_BR"

    for m in pt_PT:
        m.lang = "pt_PT"

    ans = pt_BR + pt_PT

    for m in ans:
        m.text = "Marinha sem vento, não chega a porto"

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# Romanian
def get_ro_models():
    ro_RO = [
        PiperModel(name="mihai", kind="medium", sr=22050, ns=1),
    ]

    for m in ro_RO:
        m.lang = "ro_RO"

    ans = ro_RO

    for m in ans:
        m.text = "Un foc fără lemne se stinge, o lume fără poveste moare."

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# Russian
def get_ru_models():
    ru_RU = [
        PiperModel(name="denis", kind="medium", sr=22050, ns=1),
        PiperModel(name="dmitri", kind="medium", sr=22050, ns=1),
        PiperModel(name="irina", kind="medium", sr=22050, ns=1),
        PiperModel(name="ruslan", kind="medium", sr=22050, ns=1),
    ]

    for m in ru_RU:
        m.lang = "ru_RU"

    ans = ru_RU

    for m in ans:
        m.text = "Если курица укусит, ей отрубят голову."

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# Slovak
def get_sk_models():
    sk_SK = [
        PiperModel(name="lili", kind="medium", sr=22050, ns=1),
    ]

    for m in sk_SK:
        m.lang = "sk_SK"

    ans = sk_SK

    for m in ans:
        m.text = "Kto nepozná strach, nepozná vôľu."

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# Slovenian
def get_sl_models():
    sl_SI = [
        PiperModel(name="artur", kind="medium", sr=22050, ns=1),
    ]

    for m in sl_SI:
        m.lang = "sl_SI"

    ans = sl_SI

    for m in ans:
        m.text = "Kto sa nebojí, nie je hlúpy."

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# Serbian
def get_sr_models():
    sr_RS = [
        PiperModel(name="serbski_institut", kind="medium", sr=22050, ns=2),
    ]

    for m in sr_RS:
        m.lang = "sr_RS"

    ans = sr_RS

    for m in ans:
        m.text = "Круг не може постојати без свог центра, а нација не може постојати без својих хероја."

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# Swedish
def get_sv_models():
    sv_SE = [
        PiperModel(name="lisa", kind="medium", sr=22050, ns=1),
        PiperModel(name="nst", kind="medium", sr=22050, ns=1),
    ]

    for m in sv_SE:
        m.lang = "sv_SE"

    ans = sv_SE

    for m in ans:
        m.text = "Liten skog, med många träd"

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# Swahili
def get_sw_models():
    sw_CD = [
        PiperModel(name="lanfrica", kind="medium", sr=22050, ns=1),
    ]

    for m in sw_CD:
        m.lang = "sw_CD"

    ans = sw_CD

    for m in ans:
        m.text = "Mtu mmoja hawezi kuiba mazingira."

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# Turkish
def get_tr_models():
    tr_TR = [
        PiperModel(name="dfki", kind="medium", sr=22050, ns=1),
        PiperModel(name="fahrettin", kind="medium", sr=22050, ns=1),
        PiperModel(name="fettah", kind="medium", sr=22050, ns=1),
    ]

    for m in tr_TR:
        m.lang = "tr_TR"

    ans = tr_TR

    for m in ans:
        m.text = "Bir evin duvarları, bir adamın sözü, bir kadının gülü kırılmaz"

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# Ukrainian
def get_uk_models():
    uk_UA = [
        PiperModel(name="lada", kind="x_low", sr=16000, ns=1),
        PiperModel(name="ukrainian_tts", kind="medium", sr=22050, ns=3),
    ]

    for m in uk_UA:
        m.lang = "uk_UA"

    ans = uk_UA

    for m in ans:
        m.text = "Ви не можете навчити коня, якщо не відвикнете від годівлі."

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# Vietnamese
def get_vi_models():
    vi_VN = [
        PiperModel(name="25hours_single", kind="low", sr=16000, ns=1),
        PiperModel(name="vais1000", kind="medium", sr=22050, ns=1),
        PiperModel(name="vivos", kind="x_low", sr=16000, ns=65),
    ]

    for m in vi_VN:
        m.lang = "vi_VN"

    ans = vi_VN

    for m in ans:
        m.text = "Nước cũ đào gỗ mới, sông cũ chảy nước mới"

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


# Indonesian
def get_id_models():
    id_ID = [
        PiperModel(name="news_tts", kind="medium", sr=22050, ns=1),
    ]

    for m in id_ID:
        m.lang = "id_ID"

    ans = id_ID

    for m in ans:
        m.text = "Jangan tanyakan apa yang negara bisa berikan kepadamu, tapi tanyakan apa yang bisa kamu berikan untuk negaramu."

        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


def get_en_models():
    en_gb = [
        PiperModel(name="alan", kind="low", sr=16000, ns=1),
        PiperModel(name="alan", kind="medium", sr=22050, ns=1),
        PiperModel(name="alba", kind="medium", sr=22050, ns=1),
        PiperModel(name="aru", kind="medium", sr=22050, ns=12),
        PiperModel(name="cori", kind="high", sr=22050, ns=1),
        PiperModel(name="cori", kind="medium", sr=22050, ns=1),
        PiperModel(name="jenny_dioco", kind="medium", sr=22050, ns=1),
        PiperModel(name="northern_english_male", kind="medium", sr=22050, ns=1),
        PiperModel(name="semaine", kind="medium", sr=22050, ns=4),
        PiperModel(name="southern_english_female", kind="low", sr=16000, ns=1),
        PiperModel(name="vctk", kind="medium", sr=22050, ns=109),
    ]
    en_us = [
        PiperModel(name="amy", kind="low", sr=16000, ns=1),
        PiperModel(name="amy", kind="medium", sr=22050, ns=1),
        PiperModel(name="arctic", kind="medium", sr=22050, ns=18),
        PiperModel(name="bryce", kind="medium", sr=22050, ns=1),
        PiperModel(name="danny", kind="low", sr=16000, ns=1),
        PiperModel(name="hfc_female", kind="medium", sr=22050, ns=1),
        PiperModel(name="hfc_male", kind="medium", sr=22050, ns=1),
        PiperModel(name="joe", kind="medium", sr=22050, ns=1),
        PiperModel(name="john", kind="medium", sr=22050, ns=1),
        PiperModel(name="kathleen", kind="low", sr=16000, ns=1),
        PiperModel(name="kristin", kind="medium", sr=22050, ns=1),
        PiperModel(name="kusal", kind="medium", sr=22050, ns=1),
        PiperModel(name="l2arctic", kind="medium", sr=22050, ns=24),
        PiperModel(name="lessac", kind="high", sr=22050, ns=1),
        PiperModel(name="lessac", kind="low", sr=16000, ns=1),
        PiperModel(name="lessac", kind="medium", sr=22050, ns=1),
        PiperModel(name="libritts", kind="high", sr=22050, ns=904),
        PiperModel(name="libritts_r", kind="medium", sr=22050, ns=904),
        PiperModel(name="ljspeech", kind="high", sr=22050, ns=1),
        PiperModel(name="ljspeech", kind="medium", sr=22050, ns=1),
        PiperModel(name="norman", kind="medium", sr=22050, ns=1),
        PiperModel(name="reza_ibrahim", kind="medium", sr=22050, ns=1),
        PiperModel(name="ryan", kind="high", sr=22050, ns=1),
        PiperModel(name="ryan", kind="low", sr=16000, ns=1),
        PiperModel(name="ryan", kind="medium", sr=22050, ns=1),
        PiperModel(name="sam", kind="medium", sr=22050, ns=1),
    ]

    en_gb.extend(
        [
            PiperModel(
                name="southern_english_female",
                kind="medium",
                sr=22050,
                ns=6,
                cmd="""
                   wget -qq https://huggingface.co/csukuangfj/vits-piper-en_GB-southern_english_female-medium/resolve/main/en_GB-southern_english_female-medium.onnx
                   wget -qq https://huggingface.co/csukuangfj/vits-piper-en_GB-southern_english_female-medium/resolve/main/en_GB-southern_english_female-medium.onnx.json
                   """,
                url="https://huggingface.co/csukuangfj/vits-piper-en_GB-southern_english_female-medium",
            ),
            PiperModel(
                name="southern_english_male",
                kind="medium",
                sr=22050,
                ns=8,
                cmd="""
                   wget -qq https://huggingface.co/csukuangfj/vits-piper-en_GB-southern_english_male-medium/resolve/main/en_GB-southern_english_male-medium.onnx
                   wget -qq https://huggingface.co/csukuangfj/vits-piper-en_GB-southern_english_male-medium/resolve/main/en_GB-southern_english_male-medium.onnx.json
                   """,
                url="https://huggingface.co/csukuangfj/vits-piper-en_GB-southern_english_male-medium",
            ),
        ]
    )

    en_gb += [
        PiperModel(
            name="miro",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md



                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro/resolve/main/miro_en-GB.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro/resolve/main/miro_en-GB.onnx.json

                   mv miro_en-GB.onnx en_GB-miro-high.onnx
                   mv miro_en-GB.onnx.json en_GB-miro-high.onnx.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro",
        ),
        PiperModel(
            name="dii",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-GB_dii/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_en-GB_dii" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md



                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-GB_dii/resolve/main/dii_en-GB.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-GB_dii/resolve/main/dii_en-GB.onnx.json

                   mv dii_en-GB.onnx en_GB-dii-high.onnx
                   mv dii_en-GB.onnx.json en_GB-dii-high.onnx.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_en-GB_dii",
        ),
    ]

    en_us.extend(
        [
            # https://github.com/rhasspy/piper/issues/187#issuecomment-1805709037
            # https://drive.google.com/file/d/1t2D7zP-e2flduS5duHm__UMB9RjuGqWK/view
            PiperModel(
                name="glados",
                kind="high",
                sr=22050,
                ns=1,
                cmd="""
                   wget -qq https://huggingface.co/csukuangfj/en_US-glados-high/resolve/main/en_US-glados-high.onnx
                   wget -qq https://huggingface.co/csukuangfj/en_US-glados-high/resolve/main/en_US-glados-high.onnx.json
                   wget -qq https://huggingface.co/csukuangfj/en_US-glados-high/resolve/main/README.md
                   wget -qq https://huggingface.co/csukuangfj/en_US-glados-high/resolve/main/MODEL_CARD
                   """,
                url="https://github.com/rhasspy/piper/issues/187#issuecomment-1805709037",
            ),
        ]
    )

    en_us += [
        PiperModel(
            name="miro",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-US_miro/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_en-US_miro" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md



                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-US_miro/resolve/main/miro_en-US.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_en-US_miro/resolve/main/miro_en-US.onnx.json

                   mv miro_en-US.onnx en_US-miro-high.onnx
                   mv miro_en-US.onnx.json en_US-miro-high.onnx.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_en-US_miro",
        ),
    ]

    for m in en_gb:
        m.lang = "en_GB"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    for m in en_us:
        m.lang = "en_US"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = en_gb + en_us

    for m in ans:
        m.text = "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """
        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


def get_de_models():
    de_de = [
        PiperModel(name="eva_k", kind="x_low", sr=16000, ns=1),
        PiperModel(name="karlsson", kind="low", sr=16000, ns=1),
        PiperModel(name="kerstin", kind="low", sr=16000, ns=1),
        PiperModel(name="pavoque", kind="low", sr=16000, ns=1),
        PiperModel(name="ramona", kind="low", sr=16000, ns=1),
        PiperModel(name="thorsten", kind="high", sr=22050, ns=1),
        PiperModel(name="thorsten", kind="low", sr=16000, ns=1),
        PiperModel(name="thorsten", kind="medium", sr=22050, ns=1),
        PiperModel(name="thorsten_emotional", kind="medium", sr=22050, ns=8),
        # https://github.com/rhasspy/piper/issues/187#issuecomment-2691653607
        PiperModel(
            name="glados",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados/high/de_DE-glados-high.onnx
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados/high/de_DE-glados-high.onnx.json
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados/high/MODEL_CARD
               wget -qq https://huggingface.co/csukuangfj/vits-piper-de_DE-glados-high/resolve/main/README.md
               """,
            url="https://huggingface.co/systemofapwne/piper-de-glados",
        ),
        PiperModel(
            name="glados",
            kind="low",
            sr=16000,
            ns=1,
            cmd="""
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados/low/de_DE-glados-low.onnx
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados/low/de_DE-glados-low.onnx.json
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados/low/MODEL_CARD
               wget -qq https://huggingface.co/csukuangfj/vits-piper-de_DE-glados-low/resolve/main/README.md
               """,
            url="https://huggingface.co/systemofapwne/piper-de-glados",
        ),
        PiperModel(
            name="glados",
            kind="medium",
            sr=22050,
            ns=1,
            cmd="""
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados/medium/de_DE-glados-medium.onnx
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados/medium/de_DE-glados-medium.onnx.json
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados/medium/MODEL_CARD
               wget -qq https://huggingface.co/csukuangfj/vits-piper-de_DE-glados-medium/resolve/main/README.md
               """,
            url="https://huggingface.co/systemofapwne/piper-de-glados",
        ),
        PiperModel(
            name="glados_turret",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados-turret/high/de_DE-glados-turret-high.onnx
               mv de_DE-glados-turret-high.onnx de_DE-glados_turret-high.onnx
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados-turret/high/de_DE-glados-turret-high.onnx.json
               mv de_DE-glados-turret-high.onnx.json de_DE-glados_turret-high.onnx.json
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados-turret/high/MODEL_CARD
               wget https://huggingface.co/csukuangfj/vits-piper-de_DE-glados_turret-high/resolve/main/README.md
               """,
            url="https://huggingface.co/systemofapwne/piper-de-glados",
        ),
        PiperModel(
            name="glados_turret",
            kind="low",
            sr=16000,
            ns=1,
            cmd="""
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados-turret/low/de_DE-glados-turret-low.onnx
               mv de_DE-glados-turret-low.onnx de_DE-glados_turret-low.onnx
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados-turret/low/de_DE-glados-turret-low.onnx.json
               mv de_DE-glados-turret-low.onnx.json de_DE-glados_turret-low.onnx.json
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados-turret/low/MODEL_CARD
               wget https://huggingface.co/csukuangfj/vits-piper-de_DE-glados_turret-low/resolve/main/README.md
               """,
            url="https://huggingface.co/systemofapwne/piper-de-glados",
        ),
        PiperModel(
            name="glados_turret",
            kind="medium",
            sr=22050,
            ns=1,
            cmd="""
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados-turret/medium/de_DE-glados-turret-medium.onnx
               mv de_DE-glados-turret-medium.onnx de_DE-glados_turret-medium.onnx
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados-turret/medium/de_DE-glados-turret-medium.onnx.json
               mv de_DE-glados-turret-medium.onnx.json de_DE-glados_turret-medium.onnx.json
               wget -qq https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados-turret/medium/MODEL_CARD
               wget https://huggingface.co/csukuangfj/vits-piper-de_DE-glados_turret-medium/resolve/main/README.md
               """,
            url="https://huggingface.co/systemofapwne/piper-de-glados",
        ),
    ]

    de_de += [
        PiperModel(
            name="miro",
            kind="high",
            sr=22050,
            ns=1,
            cmd="""
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_de-DE_miro/resolve/main/README.md

                   echo "\n\nSee https://huggingface.co/OpenVoiceOS/pipertts_de-DE_miro" >> README.md
                   echo "and https://github.com/OHF-Voice/piper1-gpl/discussions/27" >> README.md

                   echo "See also https://github.com/k2-fsa/sherpa-onnx/pull/2480\n\n" >> README.md
                   echo "This model is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).\n" >> README.md

                   echo "- ✅ Always free for regular (non-commercial) users  \n" >> README.md
                   echo "- ❌ Commercial use is not allowed at this time  \n" >> README.md
                   echo "- 🔄 The author may relax the restrictions in the future (e.g., allow commercial use), but will not make them stricter  \n\n" >> README.md
                   echo "**Important:** You must include this license when redistributing the model or any derivatives.\n" >> README.md



                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_de-DE_miro/resolve/main/miro_de-DE.onnx
                   wget -qq https://huggingface.co/OpenVoiceOS/pipertts_de-DE_miro/resolve/main/miro_de-DE.onnx.json

                   mv miro_de-DE.onnx de_DE-miro-high.onnx
                   mv miro_de-DE.onnx.json de_DE-miro-high.onnx.json
                   """,
            url="https://huggingface.co/OpenVoiceOS/pipertts_de-DE_miro",
        ),
    ]

    for m in de_de:
        m.lang = "de_DE"
        if m.model_name == "":
            m.model_name = f"{m.lang}-{m.name}-{m.kind}.onnx"

    ans = de_de

    for m in ans:
        m.text = "Alles hat ein Ende, nur die Wurst hat zwei."
        code = m.lang[:2]
        if m.cmd == "":
            m.cmd = f"""
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/{m.model_name}.json
            wget -qq https://huggingface.co/rhasspy/piper-voices/resolve/main/{code}/{m.lang}/{m.name}/{m.kind}/MODEL_CARD
            """

        if m.url == "":
            m.url = f"https://huggingface.co/rhasspy/piper-voices/tree/main/{code}/{m.lang}/{m.name}/{m.kind}"

    return ans


def get_all_models():
    ans = []
    ans += get_ar_models()
    ans += get_ca_models()
    ans += get_cs_models()
    ans += get_cy_models()
    ans += get_da_models()
    ans += get_de_models()
    ans += get_el_models()
    ans += get_en_models()
    ans += get_es_models()
    ans += get_fa_models()
    ans += get_fi_models()
    ans += get_fr_models()
    ans += get_hi_models()
    ans += get_id_models()
    ans += get_hu_models()
    ans += get_is_models()
    ans += get_it_models()
    ans += get_ka_models()
    ans += get_kk_models()
    ans += get_lb_models()
    ans += get_lv_models()
    ans += get_ml_models()
    ans += get_ne_models()
    ans += get_nl_models()
    ans += get_no_models()
    ans += get_pl_models()
    ans += get_pt_models()
    ans += get_ro_models()
    ans += get_ru_models()
    ans += get_sk_models()
    ans += get_sl_models()
    ans += get_sr_models()
    ans += get_sv_models()
    ans += get_sw_models()
    ans += get_tr_models()
    ans += get_uk_models()
    ans += get_vi_models()

    for i, m in enumerate(ans):
        m.index = i

    return ans


def main():
    args = get_args()
    index = args.index
    total = args.total
    assert 0 <= index < total, (index, total)

    all_model_list = get_all_models()

    print(all_model_list)

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
    for m in all_model_list:
        print(m.index, m.model_name)

    if Path("hf").is_dir():
        with open("./generate_samples.py.in") as f:
            s = f.read()
        template = environment.from_string(s)
        for m in all_model_list:
            model_dir = f"vits-piper-{m.lang}-{m.name}-{m.kind}"
            d = {
                "model": f"{model_dir}/{m.model_name}",
                "data_dir": f"{model_dir}/espeak-ng-data",
                "tokens": f"{model_dir}/tokens.txt",
                "text": m.text,
            }
            for i in range(m.ns):
                s = template.render(
                    **d,
                    sid=i,
                    output_filename=f"hf/piper/mp3/{m.lang}/{model_dir}/{i}.mp3",
                )

                with open(f"generate_samples-{model_dir}-{i}.py", "w") as f:
                    print(s, file=f)


if __name__ == "__main__":
    main()
