#!/usr/bin/env python3

"""
Generate build artifacts for the sherpa-onnx Tauri VAD+ASR desktop apps.

Outputs:
  --gen-registry    -> tauri-examples/.../src-tauri/src/model_registry.rs
  --total N --index I -> build-tauri-vad-asr.sh (Jinja2 template instantiation)

Use --target mic for the microphone variant:
  --target mic --total N --index I -> build-tauri-vad-asr-mic.sh

Usage:
  python3 scripts/tauri/generate-vad-asr.py --gen-registry
  python3 scripts/tauri/generate-vad-asr.py --total 10 --index 0
  python3 scripts/tauri/generate-vad-asr.py --target mic --total 10 --index 0
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import jinja2


@dataclass
class Model:
    model_name: str
    idx: int
    lang: str
    lang2: str
    short_name: str = ""
    cmd: str = ""
    rule_fsts: str = ""
    use_hr: bool = False
    # Rust-specific: which OfflineModelConfig field to set
    family: str = ""
    files: Dict[str, str] = field(default_factory=dict)
    model_type: str = ""
    num_threads: int = 2
    language: str = ""
    task: str = ""
    use_itn: bool = False
    src_lang: str = ""
    tgt_lang: str = ""
    use_pnc: bool = False
    tokens: str = ""


def get_models() -> List[Model]:
    return [
        Model(
            model_name="sherpa-onnx-paraformer-zh-2023-09-14",
            idx=0,
            lang="zh_en",
            lang2="Chinese,English",
            short_name="paraformer",
            family="paraformer",
            model_type="paraformer",
            files={"model": "model.int8.onnx"},
            rule_fsts="itn_zh_number.fst",cmd="""
            if [ ! -f itn_zh_number.fst ]; then
              curl -fSL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
            fi
            rm -fv $model_name/README.md
            rm -rfv $model_name/test_wavs
            rm -fv $model_name/model.onnx
            rm -fv $model_name/am.mvn $model_name/config.yaml $model_name/*.json $model_name/*.py""",
        ),
        Model(
            model_name="icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04",
            idx=1,
            lang="en_de_fr",
            lang2="English,German,French",
            short_name="zipformer",
            family="transducer",
            model_type="transducer",
            files={
                "encoder": "encoder-epoch-30-avg-4.int8.onnx",
                "decoder": "decoder-epoch-30-avg-4.onnx",
                "joiner": "joiner-epoch-30-avg-4.onnx",
            },cmd="""
            rm -fv $model_name/encoder-epoch-30-avg-4.onnx
            rm -fv $model_name/decoder-epoch-30-avg-4.int8.onnx
            rm -fv $model_name/joiner-epoch-30-avg-4.int8.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-whisper-tiny.en",
            idx=2,
            lang="en",
            lang2="English",
            short_name="whisper_tiny",
            family="whisper",
            model_type="whisper",
            language="en",
            task="transcribe",
            files={
                "encoder": "tiny.en-encoder.int8.onnx",
                "decoder": "tiny.en-decoder.int8.onnx",
            },cmd="""
            rm -fv $model_name/tiny.en-encoder.onnx
            rm -fv $model_name/tiny.en-decoder.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/requirements.txt
            rm -fv $model_name/.gitignore
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-whisper-base.en",
            idx=3,
            lang="en",
            lang2="English",
            short_name="whisper_base",
            family="whisper",
            model_type="whisper",
            language="en",
            task="transcribe",
            files={
                "encoder": "base.en-encoder.int8.onnx",
                "decoder": "base.en-decoder.int8.onnx",
            },cmd="""
            rm -fv $model_name/base.en-encoder.onnx
            rm -fv $model_name/base.en-decoder.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/requirements.txt
            rm -fv $model_name/.gitignore
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="icefall-asr-zipformer-wenetspeech-20230615",
            idx=4,
            lang="zh_en",
            lang2="Chinese,English",
            short_name="zipformer",
            family="transducer",
            model_type="transducer",
            files={
                "encoder": "encoder-epoch-12-avg-4.int8.onnx",
                "decoder": "decoder-epoch-12-avg-4.onnx",
                "joiner": "joiner-epoch-12-avg-4.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder-epoch-12-avg-4.onnx
            rm -fv $model_name/decoder-epoch-12-avg-4.int8.onnx
            rm -fv $model_name/joiner-epoch-12-avg-4.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-zipformer-multi-zh-hans-2023-9-2",
            idx=5,
            lang="zh",
            lang2="Chinese",
            short_name="zipformer",
            family="transducer",
            model_type="transducer",
            files={
                "encoder": "encoder-epoch-20-avg-1.int8.onnx",
                "decoder": "decoder-epoch-20-avg-1.onnx",
                "joiner": "joiner-epoch-20-avg-1.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder-epoch-20-avg-1.onnx
            rm -fv $model_name/decoder-epoch-20-avg-1.int8.onnx
            rm -fv $model_name/joiner-epoch-20-avg-1.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-ctc-en-citrinet-512",
            idx=6,
            lang="en",
            lang2="English",
            short_name="nemo_ctc",
            family="nemo_ctc",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k",
            idx=7,
            lang="multi",
            lang2="Multi-language",
            short_name="nemo_conformer",
            family="nemo_ctc",
            files={"model": "model.onnx"},cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-fast-conformer-ctc-en-24500",
            idx=8,
            lang="en",
            lang2="English",
            short_name="nemo_conformer",
            family="nemo_ctc",
            files={"model": "model.onnx"},cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-fast-conformer-ctc-en-de-es-fr-14288",
            idx=9,
            lang="en_de_es_fr",
            lang2="English,German,Spanish,French",
            short_name="nemo_conformer",
            family="nemo_ctc",
            files={"model": "model.onnx"},cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-fast-conformer-ctc-es-1424",
            idx=10,
            lang="es",
            lang2="Spanish",
            short_name="nemo_conformer",
            family="nemo_ctc",
            files={"model": "model.onnx"},cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04",
            idx=11,
            lang="zh",
            lang2="Chinese",
            short_name="telespeech",
            family="telespeech_ctc",
            model_type="telespeech_ctc",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-zipformer-thai-2024-06-20",
            idx=12,
            lang="th",
            lang2="Thai",
            short_name="zipformer",
            family="transducer",
            model_type="transducer",
            files={
                "encoder": "encoder-epoch-12-avg-5.int8.onnx",
                "decoder": "decoder-epoch-12-avg-5.onnx",
                "joiner": "joiner-epoch-12-avg-5.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder-epoch-12-avg-5.onnx
            rm -fv $model_name/decoder-epoch-12-avg-5.int8.onnx
            rm -fv $model_name/joiner-epoch-12-avg-5.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-zipformer-korean-2024-06-24",
            idx=13,
            lang="ko",
            lang2="Korean",
            short_name="zipformer",
            family="transducer",
            model_type="transducer",
            files={
                "encoder": "encoder-epoch-99-avg-1.int8.onnx",
                "decoder": "decoder-epoch-99-avg-1.onnx",
                "joiner": "joiner-epoch-99-avg-1.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder-epoch-99-avg-1.onnx
            rm -fv $model_name/decoder-epoch-99-avg-1.int8.onnx
            rm -fv $model_name/joiner-epoch-99-avg-1.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-paraformer-zh-small-2024-03-09",
            idx=14,
            lang="zh_en",
            lang2="Chinese,English",
            short_name="small_paraformer",
            family="paraformer",
            model_type="paraformer",
            files={"model": "model.int8.onnx"},
            rule_fsts="itn_zh_number.fst",cmd="""
            if [ ! -f itn_zh_number.fst ]; then
              curl -fSL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
            fi
            rm -fv $model_name/model.onnx
            rm -fv $model_name/README.md
            rm -rfv $model_name/test_wavs""",
        ),
        Model(
            model_name="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
            idx=15,
            lang="zh_en_ko_ja_yue",
            lang2="中英粤日韩",
            short_name="sense_voice_2024_07_17_int8",
            family="sense_voice",
            use_itn=True,
            files={"model": "model.int8.onnx"},
            use_hr=True,cmd="""
            rm -rfv $model_name/test_wavs
            rm -fv $model_name/*.py""",
        ),
        Model(
            model_name="sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
            idx=16,
            lang="ja",
            lang2="Japanese",
            short_name="zipformer",
            family="transducer",
            model_type="transducer",
            files={
                "encoder": "encoder-epoch-99-avg-1.int8.onnx",
                "decoder": "decoder-epoch-99-avg-1.onnx",
                "joiner": "joiner-epoch-99-avg-1.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder-epoch-99-avg-1.onnx
            rm -fv $model_name/decoder-epoch-99-avg-1.int8.onnx
            rm -fv $model_name/joiner-epoch-99-avg-1.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-zipformer-ru-2024-09-18",
            idx=17,
            lang="ru",
            lang2="Russian",
            short_name="zipformer",
            family="transducer",
            model_type="transducer",
            files={
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.onnx",
                "joiner": "joiner.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder.onnx
            rm -fv $model_name/joiner.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-small-zipformer-ru-2024-09-18",
            idx=18,
            lang="ru",
            lang2="Russian",
            short_name="small_zipformer",
            family="transducer",
            model_type="transducer",
            files={
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.onnx",
                "joiner": "joiner.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder.onnx
            rm -fv $model_name/joiner.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24",
            idx=19,
            lang="ru",
            lang2="Russian",
            short_name="nemo_ctc",
            family="nemo_ctc",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24",
            idx=20,
            lang="ru",
            lang2="Russian",
            short_name="nemo_transducer",
            family="transducer",
            model_type="nemo_transducer",
            files={
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.onnx",
                "joiner": "joiner.onnx",
            },cmd="""
            rm -fv $model_name/encoder.onnx
            rm -fv $model_name/joiner.int8.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-moonshine-tiny-en-int8",
            idx=21,
            lang="en",
            lang2="English",
            short_name="moonshine_tiny",
            family="moonshine",
            files={
                "preprocessor": "preprocess.onnx",
                "encoder": "encode.int8.onnx",
                "uncached_decoder": "uncached_decode.int8.onnx",
                "cached_decoder": "cached_decode.int8.onnx",
            },cmd="""
            rm -fv $model_name/preprocess.onnx
            rm -fv $model_name/encode.onnx
            rm -fv $model_name/uncached_decode.onnx
            rm -fv $model_name/cached_decode.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-moonshine-base-en-int8",
            idx=22,
            lang="en",
            lang2="English",
            short_name="moonshine_base",
            family="moonshine",
            files={
                "preprocessor": "preprocess.onnx",
                "encoder": "encode.int8.onnx",
                "uncached_decoder": "uncached_decode.int8.onnx",
                "cached_decoder": "cached_decode.int8.onnx",
            },cmd="""
            rm -fv $model_name/preprocess.onnx
            rm -fv $model_name/encode.onnx
            rm -fv $model_name/uncached_decode.onnx
            rm -fv $model_name/cached_decode.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-zipformer-zh-en-2023-11-22",
            idx=23,
            lang="zh_en",
            lang2="Chinese,English",
            short_name="zipformer",
            family="transducer",
            model_type="transducer",
            files={
                "encoder": "encoder-epoch-34-avg-19.int8.onnx",
                "decoder": "decoder-epoch-34-avg-19.onnx",
                "joiner": "joiner-epoch-34-avg-19.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder-epoch-34-avg-19.onnx
            rm -fv $model_name/decoder-epoch-34-avg-19.int8.onnx
            rm -fv $model_name/joiner-epoch-34-avg-19.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16",
            idx=24,
            lang="zh_en",
            lang2="Chinese,English",
            short_name="fire_red_asr",
            family="fire_red_asr",
            files={
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder.onnx
            rm -fv $model_name/decoder.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02",
            idx=25,
            lang="multi",
            lang2="Multi-language",
            short_name="dolphin",
            family="dolphin",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-zipformer-vi-int8-2025-04-20",
            idx=26,
            lang="vi",
            lang2="Vietnamese",
            short_name="zipformer",
            family="transducer",
            model_type="transducer",
            files={
                "encoder": "encoder-epoch-12-avg-8.int8.onnx",
                "decoder": "decoder-epoch-12-avg-8.onnx",
                "joiner": "joiner-epoch-12-avg-8.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder-epoch-12-avg-8.onnx
            rm -fv $model_name/decoder-epoch-12-avg-8.int8.onnx
            rm -fv $model_name/joiner-epoch-12-avg-8.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19",
            idx=27,
            lang="ru",
            lang2="Russian",
            short_name="nemo_ctc_v2",
            family="nemo_ctc",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19",
            idx=28,
            lang="ru",
            lang2="Russian",
            short_name="nemo_transducer_v2",
            family="transducer",
            model_type="nemo_transducer",
            files={
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.onnx",
                "joiner": "joiner.onnx",
            },cmd="""
            rm -fv $model_name/encoder.onnx
            rm -fv $model_name/joiner.int8.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-zipformer-ru-int8-2025-04-20",
            idx=29,
            lang="ru",
            lang2="Russian",
            short_name="zipformer_int8",
            family="transducer",
            model_type="transducer",
            files={
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.onnx",
                "joiner": "joiner.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder.onnx
            rm -fv $model_name/joiner.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
            idx=30,
            lang="en",
            lang2="English",
            short_name="parakeet_tdt",
            family="transducer",
            model_type="nemo_transducer",
            files={
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.int8.onnx",
                "joiner": "joiner.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder.onnx
            rm -fv $model_name/decoder.onnx
            rm -fv $model_name/joiner.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03",
            idx=31,
            lang="zh",
            lang2="Chinese",
            short_name="zipformer_ctc",
            family="zipformer_ctc",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8",
            idx=32,
            lang="en_es_de_fr",
            lang2="English,Spanish,German,French",
            short_name="canary",
            family="canary",
            src_lang="en",
            tgt_lang="en",
            use_pnc=True,
            files={
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder.onnx
            rm -fv $model_name/decoder.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8",
            idx=33,
            lang="en",
            lang2="English",
            short_name="parakeet_tdt_ctc",
            family="nemo_ctc",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8",
            idx=34,
            lang="ja",
            lang2="Japanese",
            short_name="parakeet_ja",
            family="nemo_ctc",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-transducer-stt_pt_fastconformer_hybrid_large_pc-int8",
            idx=35,
            lang="pt",
            lang2="Portuguese",
            short_name="nemo_transducer_pt",
            family="transducer",
            model_type="nemo_transducer",
            files={
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.int8.onnx",
                "joiner": "joiner.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder.onnx
            rm -fv $model_name/decoder.onnx
            rm -fv $model_name/joiner.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-stt_pt_fastconformer_hybrid_large_pc-int8",
            idx=36,
            lang="pt",
            lang2="Portuguese",
            short_name="nemo_ctc_pt",
            family="nemo_ctc",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-transducer-stt_de_fastconformer_hybrid_large_pc-int8",
            idx=37,
            lang="de",
            lang2="German",
            short_name="nemo_transducer_de",
            family="transducer",
            model_type="nemo_transducer",
            files={
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.int8.onnx",
                "joiner": "joiner.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder.onnx
            rm -fv $model_name/decoder.onnx
            rm -fv $model_name/joiner.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-stt_de_fastconformer_hybrid_large_pc-int8",
            idx=38,
            lang="de",
            lang2="German",
            short_name="nemo_ctc_de",
            family="nemo_ctc",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-zipformer-ctc-small-zh-int8-2025-07-16",
            idx=39,
            lang="zh",
            lang2="Chinese",
            short_name="zipformer_ctc_small",
            family="zipformer_ctc",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            idx=40,
            lang="en",
            lang2="English",
            short_name="parakeet_tdt_v3",
            family="transducer",
            model_type="nemo_transducer",
            files={
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.int8.onnx",
                "joiner": "joiner.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder.onnx
            rm -fv $model_name/decoder.onnx
            rm -fv $model_name/joiner.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09",
            idx=41,
            lang="zh_en_ko_ja_yue",
            lang2="中英粤日韩",
            short_name="sense_voice_2025_09_09_int8",
            family="sense_voice",
            use_itn=True,
            files={"model": "model.int8.onnx"},
            use_hr=True,cmd="""
            rm -rfv $model_name/test_wavs
            rm -fv $model_name/*.py""",
        ),
        Model(
            model_name="sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10",
            idx=42,
            lang="zh_en_yue",
            lang2="Chinese,English,Cantonese",
            short_name="wenet_yue",
            family="wenet_ctc",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-paraformer-zh-int8-2025-10-07",
            idx=43,
            lang="zh_en",
            lang2="Chinese,English",
            short_name="paraformer_v2",
            family="paraformer",
            model_type="paraformer",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -fv $model_name/README.md
            rm -rfv $model_name/test_wavs""",
        ),
        Model(
            model_name="sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12",
            idx=44,
            lang="multi",
            lang2="1600 languages",
            short_name="omnilingual",
            family="omnilingual",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-medasr-ctc-en-int8-2025-12-25",
            idx=45,
            lang="en",
            lang2="English",
            short_name="medasr",
            family="medasr",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-funasr-nano-int8-2025-12-30",
            idx=46,
            lang="multi",
            lang2="31 languages",
            short_name="funasr_nano",
            family="funasr_nano",
            num_threads=3,
            tokens="",
            files={
                "encoder_adaptor": "encoder_adaptor.int8.onnx",
                "llm": "llm.int8.onnx",
                "embedding": "embedding.int8.onnx",
                "tokenizer": "Qwen3-0.6B",
            },cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03",
            idx=47,
            lang="zh",
            lang2="Chinese (Wu dialect)",
            short_name="wenet_wu",
            family="wenet_ctc",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-2026-02-03",
            idx=48,
            lang="zh",
            lang2="Chinese (Wu dialect)",
            short_name="wenet_wu_fp32",
            family="wenet_ctc",
            files={"model": "model.onnx"},cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
            idx=49,
            lang="vi",
            lang2="Vietnamese",
            short_name="zipformer_vi",
            family="transducer",
            model_type="transducer",
            files={
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.onnx",
                "joiner": "joiner.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder.onnx
            rm -fv $model_name/joiner.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25",
            idx=50,
            lang="zh_en",
            lang2="Chinese,English",
            short_name="fire_red_asr2_ctc",
            family="fire_red_asr_ctc",
            files={"model": "model.int8.onnx"},cmd="""
            rm -fv $model_name/model.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        # Moonshine v2 models (51-60)
        Model(
            model_name="sherpa-onnx-moonshine-tiny-ko-quantized-2026-02-27",
            idx=51,
            lang="ko",
            lang2="Korean",
            short_name="moonshine_v2_tiny_ko",
            family="moonshine_v2",
            files={
                "encoder": "encoder_model.ort",
                "merged_decoder": "decoder_model_merged.ort",
            },cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-moonshine-tiny-ja-quantized-2026-02-27",
            idx=52,
            lang="ja",
            lang2="Japanese",
            short_name="moonshine_v2_tiny_ja",
            family="moonshine_v2",
            files={
                "encoder": "encoder_model.ort",
                "merged_decoder": "decoder_model_merged.ort",
            },cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27",
            idx=53,
            lang="en",
            lang2="English",
            short_name="moonshine_v2_tiny_en",
            family="moonshine_v2",
            files={
                "encoder": "encoder_model.ort",
                "merged_decoder": "decoder_model_merged.ort",
            },cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-moonshine-base-zh-quantized-2026-02-27",
            idx=54,
            lang="zh",
            lang2="Chinese",
            short_name="moonshine_v2_base_zh",
            family="moonshine_v2",
            files={
                "encoder": "encoder_model.ort",
                "merged_decoder": "decoder_model_merged.ort",
            },cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-moonshine-base-vi-quantized-2026-02-27",
            idx=55,
            lang="vi",
            lang2="Vietnamese",
            short_name="moonshine_v2_base_vi",
            family="moonshine_v2",
            files={
                "encoder": "encoder_model.ort",
                "merged_decoder": "decoder_model_merged.ort",
            },cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-moonshine-base-uk-quantized-2026-02-27",
            idx=56,
            lang="uk",
            lang2="Ukrainian",
            short_name="moonshine_v2_base_uk",
            family="moonshine_v2",
            files={
                "encoder": "encoder_model.ort",
                "merged_decoder": "decoder_model_merged.ort",
            },cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-moonshine-base-ja-quantized-2026-02-27",
            idx=57,
            lang="ja",
            lang2="Japanese",
            short_name="moonshine_v2_base_ja",
            family="moonshine_v2",
            files={
                "encoder": "encoder_model.ort",
                "merged_decoder": "decoder_model_merged.ort",
            },cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-moonshine-base-es-quantized-2026-02-27",
            idx=58,
            lang="es",
            lang2="Spanish",
            short_name="moonshine_v2_base_es",
            family="moonshine_v2",
            files={
                "encoder": "encoder_model.ort",
                "merged_decoder": "decoder_model_merged.ort",
            },cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-moonshine-base-en-quantized-2026-02-27",
            idx=59,
            lang="en",
            lang2="English",
            short_name="moonshine_v2_base_en",
            family="moonshine_v2",
            files={
                "encoder": "encoder_model.ort",
                "merged_decoder": "decoder_model_merged.ort",
            },cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-moonshine-base-ar-quantized-2026-02-27",
            idx=60,
            lang="ar",
            lang2="Arabic",
            short_name="moonshine_v2_base_ar",
            family="moonshine_v2",
            files={
                "encoder": "encoder_model.ort",
                "merged_decoder": "decoder_model_merged.ort",
            },cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
            idx=61,
            lang="multi",
            lang2="52 languages",
            short_name="qwen3_asr",
            family="qwen3_asr",
            num_threads=3,
            tokens="",
            files={
                "conv_frontend": "conv_frontend.onnx",
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.int8.onnx",
                "tokenizer": "tokenizer",
            },cmd="""
            rm -rf $model_name/test_wavs
            rm -fv $model_name/README.md""",
        ),
        Model(
            model_name="sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming",
            idx=62,
            lang="en",
            lang2="English",
            short_name="parakeet_unified_en_0.6b_int8",
            family="transducer",
            model_type="nemo_transducer",
            files={
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.int8.onnx",
                "joiner": "joiner.int8.onnx",
            },cmd="""
            rm -fv $model_name/encoder.onnx
            rm -fv $model_name/decoder.onnx
            rm -fv $model_name/joiner.onnx
            rm -rf $model_name/test_wavs
            rm -fv $model_name/*.py
            rm -fv $model_name/README.md""",
        ),
    ]


def gen_model_registry(models: List[Model], output_path: Path):
    """Generate model_registry.rs from model definitions."""

    lines = []
    lines.append("// Auto-generated by generate-vad-asr.py. DO NOT EDIT.")
    lines.append("")
    lines.append("use std::path::Path;")
    lines.append("")
    lines.append("use sherpa_onnx::{")
    lines.append("    OfflineCanaryModelConfig, OfflineDolphinModelConfig,")
    lines.append("    OfflineFireRedAsrCtcModelConfig, OfflineFireRedAsrModelConfig,")
    lines.append("    OfflineFunASRNanoModelConfig, OfflineMedAsrCtcModelConfig,")
    lines.append("    OfflineMoonshineModelConfig, OfflineNemoEncDecCtcModelConfig,")
    lines.append("    OfflineOmnilingualAsrCtcModelConfig, OfflineParaformerModelConfig,")
    lines.append("    OfflineQwen3ASRModelConfig, OfflineRecognizerConfig,")
    lines.append("    OfflineSenseVoiceModelConfig, OfflineTransducerModelConfig,")
    lines.append("    OfflineWenetCtcModelConfig, OfflineWhisperModelConfig,")
    lines.append("    OfflineZipformerCtcModelConfig,")
    lines.append("};")
    lines.append("")
    lines.append("pub fn get_model_config(model_type: u32, model_dir: &Path) -> Option<OfflineRecognizerConfig> {")
    lines.append("    let p = |sub: &str| -> Option<String> {")
    lines.append("        let path = model_dir.join(sub);")
    lines.append("        if !path.exists() { return None; }")
    lines.append("        Some(path.to_str()?.to_string())")
    lines.append("    };")
    lines.append("    match model_type {")

    for model in sorted(models, key=lambda m: m.idx):
        lines.append(f"        {model.idx} => {{")
        lines.append(f'            // {model.model_name}')
        lines.append(f"            let mut config = OfflineRecognizerConfig::default();")

        family = model.family
        files = model.files

        if family == "paraformer":
            lines.append(f'            config.model_config.paraformer = OfflineParaformerModelConfig {{')
            lines.append(f'                model: p("{files["model"]}"),')
            lines.append(f"            }};")
        elif family == "transducer":
            lines.append(f'            config.model_config.transducer = OfflineTransducerModelConfig {{')
            lines.append(f'                encoder: p("{files["encoder"]}"),')
            lines.append(f'                decoder: p("{files["decoder"]}"),')
            lines.append(f'                joiner: p("{files["joiner"]}"),')
            lines.append(f"            }};")
        elif family == "whisper":
            lines.append(f'            config.model_config.whisper = OfflineWhisperModelConfig {{')
            lines.append(f'                encoder: p("{files["encoder"]}"),')
            lines.append(f'                decoder: p("{files["decoder"]}"),')
            if model.language:
                lines.append(f'                language: Some("{model.language}".into()),')
            if model.task:
                lines.append(f'                task: Some("{model.task}".into()),')
            lines.append(f"                ..Default::default()")
            lines.append(f"            }};")
        elif family == "nemo_ctc":
            lines.append(f'            config.model_config.nemo_ctc = OfflineNemoEncDecCtcModelConfig {{')
            lines.append(f'                model: p("{files["model"]}"),')
            lines.append(f"            }};")
        elif family == "sense_voice":
            lines.append(f'            config.model_config.sense_voice = OfflineSenseVoiceModelConfig {{')
            lines.append(f'                model: p("{files["model"]}"),')
            if model.use_itn:
                lines.append(f'                use_itn: true,')
            lines.append(f"                ..Default::default()")
            lines.append(f"            }};")
        elif family == "telespeech_ctc":
            lines.append(f'            config.model_config.telespeech_ctc = p("{files["model"]}");')
        elif family == "moonshine":
            lines.append(f'            config.model_config.moonshine = OfflineMoonshineModelConfig {{')
            lines.append(f'                preprocessor: p("{files["preprocessor"]}"),')
            lines.append(f'                encoder: p("{files["encoder"]}"),')
            lines.append(f'                uncached_decoder: p("{files["uncached_decoder"]}"),')
            lines.append(f'                cached_decoder: p("{files["cached_decoder"]}"),')
            lines.append(f"                ..Default::default()")
            lines.append(f"            }};")
        elif family == "moonshine_v2":
            lines.append(f'            config.model_config.moonshine = OfflineMoonshineModelConfig {{')
            lines.append(f'                encoder: p("{files["encoder"]}"),')
            lines.append(f'                merged_decoder: p("{files["merged_decoder"]}"),')
            lines.append(f"                ..Default::default()")
            lines.append(f"            }};")
        elif family == "fire_red_asr":
            lines.append(f'            config.model_config.fire_red_asr = OfflineFireRedAsrModelConfig {{')
            lines.append(f'                encoder: p("{files["encoder"]}"),')
            lines.append(f'                decoder: p("{files["decoder"]}"),')
            lines.append(f"            }};")
        elif family == "dolphin":
            lines.append(f'            config.model_config.dolphin = OfflineDolphinModelConfig {{')
            lines.append(f'                model: p("{files["model"]}"),')
            lines.append(f"            }};")
        elif family == "zipformer_ctc":
            lines.append(f'            config.model_config.zipformer_ctc = OfflineZipformerCtcModelConfig {{')
            lines.append(f'                model: p("{files["model"]}"),')
            lines.append(f"            }};")
        elif family == "canary":
            lines.append(f'            config.model_config.canary = OfflineCanaryModelConfig {{')
            lines.append(f'                encoder: p("{files["encoder"]}"),')
            lines.append(f'                decoder: p("{files["decoder"]}"),')
            if model.src_lang:
                lines.append(f'                src_lang: Some("{model.src_lang}".into()),')
            if model.tgt_lang:
                lines.append(f'                tgt_lang: Some("{model.tgt_lang}".into()),')
            if model.use_pnc:
                lines.append(f'                use_pnc: true,')
            lines.append(f"                ..Default::default()")
            lines.append(f"            }};")
        elif family == "wenet_ctc":
            lines.append(f'            config.model_config.wenet_ctc = OfflineWenetCtcModelConfig {{')
            lines.append(f'                model: p("{files["model"]}"),')
            lines.append(f"            }};")
        elif family == "omnilingual":
            lines.append(f'            config.model_config.omnilingual = OfflineOmnilingualAsrCtcModelConfig {{')
            lines.append(f'                model: p("{files["model"]}"),')
            lines.append(f"            }};")
        elif family == "medasr":
            lines.append(f'            config.model_config.medasr = OfflineMedAsrCtcModelConfig {{')
            lines.append(f'                model: p("{files["model"]}"),')
            lines.append(f"            }};")
        elif family == "funasr_nano":
            lines.append(f'            config.model_config.funasr_nano = OfflineFunASRNanoModelConfig {{')
            lines.append(f'                encoder_adaptor: p("{files["encoder_adaptor"]}"),')
            lines.append(f'                llm: p("{files["llm"]}"),')
            lines.append(f'                embedding: p("{files["embedding"]}"),')
            lines.append(f'                tokenizer: p("{files["tokenizer"]}"),')
            lines.append(f"                ..Default::default()")
            lines.append(f"            }};")
        elif family == "fire_red_asr_ctc":
            lines.append(f'            config.model_config.fire_red_asr_ctc = OfflineFireRedAsrCtcModelConfig {{')
            lines.append(f'                model: p("{files["model"]}"),')
            lines.append(f"            }};")
        elif family == "qwen3_asr":
            lines.append(f'            config.model_config.qwen3_asr = OfflineQwen3ASRModelConfig {{')
            lines.append(f'                conv_frontend: p("{files["conv_frontend"]}"),')
            lines.append(f'                encoder: p("{files["encoder"]}"),')
            lines.append(f'                decoder: p("{files["decoder"]}"),')
            lines.append(f'                tokenizer: p("{files["tokenizer"]}"),')
            lines.append(f"                ..Default::default()")
            lines.append(f"            }};")

        # Set common fields
        if model.tokens:
            pass
        else:
            lines.append(f'            config.model_config.tokens = p("tokens.txt");')

        if model.model_type:
            lines.append(f'            config.model_config.model_type = Some("{model.model_type}".into());')

        lines.append(f'            config.model_config.num_threads = {model.num_threads};')
        lines.append(f'            config.model_config.debug = true;')

        if model.rule_fsts:
            lines.append(f'            config.rule_fsts = p("{model.rule_fsts}");')

        # HR files (lexicon.txt, rule.fst) are resolved at runtime from
        # resource_dir(), not from model_dir, so we don't emit them here.

        lines.append(f"            Some(config)")
        lines.append(f"        }}")
        lines.append("")

    lines.append("        _ => None,")
    lines.append("    }")
    lines.append("}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Generated {output_path}")


def gen_build_script(models: List[Model], output_path: Path, template_name: str = "build-tauri-vad-asr.sh.in"):
    """Render the Jinja2 template with the given model list."""
    environment = jinja2.Environment()
    template_path = Path(__file__).parent / template_name
    with open(template_path) as f:
        template = environment.from_string(f.read())

    rendered = template.render(model_list=models)

    output_path.write_text(rendered)
    output_path.chmod(0o755)
    print(f"Generated {output_path} ({len(models)} models)")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-registry", action="store_true", help="Generate model_registry.rs")
    parser.add_argument("--target", choices=["file", "mic"], default="file", help="Which app variant to build (file or mic)")
    parser.add_argument(
        "--total",
        type=int,
        default=1,
        help="Number of shards (runners)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of this shard (0-based)",
    )
    return parser.parse_args()


def main():
    args = get_args()
    all_models = get_models()

    repo_root = Path(__file__).parent.parent.parent
    tauri_src = repo_root / "tauri-examples" / "non-streaming-speech-recognition-from-file" / "src-tauri"
    scripts_dir = repo_root / "scripts" / "tauri"

    if args.gen_registry:
        gen_model_registry(all_models, tauri_src / "src" / "model_registry.rs")
        mic_tauri_src = repo_root / "tauri-examples" / "non-streaming-speech-recognition-from-microphone" / "src-tauri"
        gen_model_registry(all_models, mic_tauri_src / "src" / "model_registry.rs")
        return

    # Shard models using same logic as APK pattern
    total = args.total
    index = args.index
    assert 0 <= index < total, (index, total)

    num_models = len(all_models)
    num_per_runner = num_models // total
    if num_per_runner <= 0:
        num_per_runner = 0

    start = index * num_per_runner
    end = start + num_per_runner

    remaining = num_models - total * num_per_runner

    model_list = all_models[start:end]
    if index < remaining:
        s = total * num_per_runner + index
        model_list.append(all_models[s])

    print(f"Shard {index}/{total}: models {[m.idx for m in model_list]}")

    if args.target == "mic":
        gen_build_script(model_list, scripts_dir / "build-tauri-vad-asr-mic.sh", "build-tauri-vad-asr-mic.sh.in")
    else:
        gen_build_script(model_list, scripts_dir / "build-tauri-vad-asr.sh")


if __name__ == "__main__":
    main()
