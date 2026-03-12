#!/usr/bin/env python3
"""
FunASR vs sherpa-onnx fork (is_final) Benchmark

用同一组 73 条测试语料，对比 FunASR 流式 Paraformer 和
sherpa-onnx fork (带 is_final) 的识别结果与 CER。

FunASR 使用 PyTorch 原版流式 Paraformer（paraformer-zh-streaming），
sherpa-onnx 使用 ONNX INT8 量化版本。

两者使用相同的流式解码方式（0.6s chunk），
FunASR 最后一个 chunk 传 is_final=True，
sherpa-onnx fork 调用 SetOption("is_final", "true") 后再解码。
"""

import ctypes
import json
import os
import re
import struct
import sys
import time
import warnings
import wave
from pathlib import Path

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────────
SHERPA_LIB = Path(__file__).parent.parent / "build-is-final" / "lib" / "libsherpa-onnx-c-api.dylib"
ORT_LIB = Path(__file__).parent.parent / "build-is-final" / "_deps" / "onnxruntime-src" / "lib" / "libonnxruntime.dylib"

MODEL_DIR = Path.home() / "Library" / "Application Support" / "Nano Typeless" / "models" / "sherpa-onnx-streaming-paraformer-bilingual-zh-en"
ENCODER = MODEL_DIR / "encoder.int8.onnx"
DECODER = MODEL_DIR / "decoder.int8.onnx"
TOKENS = MODEL_DIR / "tokens.txt"

TYPELESS_ROOT = Path.home() / "Github" / "typeless"
FIXTURES = TYPELESS_ROOT / "Tests" / "fixtures"
CORPUS_JSON = FIXTURES / "corpus.json"
REAL_MANIFEST = FIXTURES / "real_manifest.json"


# ──────────────────────────────────────────────────
# sherpa-onnx ctypes 结构体（与 test_is_final.py 相同）
# ──────────────────────────────────────────────────
class SherpaOnnxOnlineTransducerModelConfig(ctypes.Structure):
    _fields_ = [("encoder", ctypes.c_char_p), ("decoder", ctypes.c_char_p), ("joiner", ctypes.c_char_p)]

class SherpaOnnxOnlineParaformerModelConfig(ctypes.Structure):
    _fields_ = [("encoder", ctypes.c_char_p), ("decoder", ctypes.c_char_p)]

class SherpaOnnxOnlineZipformer2CtcModelConfig(ctypes.Structure):
    _fields_ = [("model", ctypes.c_char_p)]

class SherpaOnnxOnlineNemoCtcModelConfig(ctypes.Structure):
    _fields_ = [("model", ctypes.c_char_p)]

class SherpaOnnxOnlineToneCtcModelConfig(ctypes.Structure):
    _fields_ = [("model", ctypes.c_char_p)]

class SherpaOnnxOnlineModelConfig(ctypes.Structure):
    _fields_ = [
        ("transducer", SherpaOnnxOnlineTransducerModelConfig),
        ("paraformer", SherpaOnnxOnlineParaformerModelConfig),
        ("zipformer2_ctc", SherpaOnnxOnlineZipformer2CtcModelConfig),
        ("tokens", ctypes.c_char_p), ("num_threads", ctypes.c_int32),
        ("provider", ctypes.c_char_p), ("debug", ctypes.c_int32),
        ("model_type", ctypes.c_char_p), ("modeling_unit", ctypes.c_char_p),
        ("bpe_vocab", ctypes.c_char_p), ("tokens_buf", ctypes.c_char_p),
        ("tokens_buf_size", ctypes.c_int32),
        ("nemo_ctc", SherpaOnnxOnlineNemoCtcModelConfig),
        ("t_one_ctc", SherpaOnnxOnlineToneCtcModelConfig),
    ]

class SherpaOnnxFeatureConfig(ctypes.Structure):
    _fields_ = [("sample_rate", ctypes.c_int32), ("feature_dim", ctypes.c_int32)]

class SherpaOnnxOnlineCtcFstDecoderConfig(ctypes.Structure):
    _fields_ = [("graph", ctypes.c_char_p), ("max_active", ctypes.c_int32)]

class SherpaOnnxHomophoneReplacerConfig(ctypes.Structure):
    _fields_ = [("dict_dir", ctypes.c_char_p), ("lexicon", ctypes.c_char_p), ("rule_fsts", ctypes.c_char_p)]

class SherpaOnnxOnlineRecognizerConfig(ctypes.Structure):
    _fields_ = [
        ("feat_config", SherpaOnnxFeatureConfig),
        ("model_config", SherpaOnnxOnlineModelConfig),
        ("decoding_method", ctypes.c_char_p), ("max_active_paths", ctypes.c_int32),
        ("enable_endpoint", ctypes.c_int32),
        ("rule1_min_trailing_silence", ctypes.c_float),
        ("rule2_min_trailing_silence", ctypes.c_float),
        ("rule3_min_utterance_length", ctypes.c_float),
        ("hotwords_file", ctypes.c_char_p), ("hotwords_score", ctypes.c_float),
        ("ctc_fst_decoder_config", SherpaOnnxOnlineCtcFstDecoderConfig),
        ("rule_fsts", ctypes.c_char_p), ("rule_fars", ctypes.c_char_p),
        ("blank_penalty", ctypes.c_float),
        ("hotwords_buf", ctypes.c_char_p), ("hotwords_buf_size", ctypes.c_int32),
        ("hr", SherpaOnnxHomophoneReplacerConfig),
    ]

class SherpaOnnxOnlineRecognizerResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p), ("tokens", ctypes.c_char_p),
        ("tokens_arr", ctypes.POINTER(ctypes.c_char_p)),
        ("timestamps", ctypes.POINTER(ctypes.c_float)),
        ("count", ctypes.c_int32), ("json", ctypes.c_char_p),
    ]


# ──────────────────────────────────────────────────
# sherpa-onnx C API 加载
# ──────────────────────────────────────────────────
def load_sherpa_lib():
    if ORT_LIB.exists():
        ctypes.CDLL(str(ORT_LIB))
    lib = ctypes.CDLL(str(SHERPA_LIB))

    lib.SherpaOnnxCreateOnlineRecognizer.argtypes = [ctypes.POINTER(SherpaOnnxOnlineRecognizerConfig)]
    lib.SherpaOnnxCreateOnlineRecognizer.restype = ctypes.c_void_p
    lib.SherpaOnnxDestroyOnlineRecognizer.argtypes = [ctypes.c_void_p]
    lib.SherpaOnnxDestroyOnlineRecognizer.restype = None
    lib.SherpaOnnxCreateOnlineStream.argtypes = [ctypes.c_void_p]
    lib.SherpaOnnxCreateOnlineStream.restype = ctypes.c_void_p
    lib.SherpaOnnxDestroyOnlineStream.argtypes = [ctypes.c_void_p]
    lib.SherpaOnnxDestroyOnlineStream.restype = None
    lib.SherpaOnnxOnlineStreamAcceptWaveform.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    lib.SherpaOnnxOnlineStreamAcceptWaveform.restype = None
    lib.SherpaOnnxOnlineStreamInputFinished.argtypes = [ctypes.c_void_p]
    lib.SherpaOnnxOnlineStreamInputFinished.restype = None
    lib.SherpaOnnxOnlineStreamSetOption.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
    ]
    lib.SherpaOnnxOnlineStreamSetOption.restype = None
    lib.SherpaOnnxIsOnlineStreamReady.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.SherpaOnnxIsOnlineStreamReady.restype = ctypes.c_int32
    lib.SherpaOnnxDecodeOnlineStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.SherpaOnnxDecodeOnlineStream.restype = None
    lib.SherpaOnnxGetOnlineStreamResult.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.SherpaOnnxGetOnlineStreamResult.restype = ctypes.POINTER(SherpaOnnxOnlineRecognizerResult)
    lib.SherpaOnnxDestroyOnlineRecognizerResult.argtypes = [ctypes.POINTER(SherpaOnnxOnlineRecognizerResult)]
    lib.SherpaOnnxDestroyOnlineRecognizerResult.restype = None
    return lib


def create_sherpa_recognizer(lib):
    config = SherpaOnnxOnlineRecognizerConfig()
    ctypes.memset(ctypes.byref(config), 0, ctypes.sizeof(config))
    config.feat_config.sample_rate = 16000
    config.feat_config.feature_dim = 80
    config.model_config.paraformer.encoder = str(ENCODER).encode("utf-8")
    config.model_config.paraformer.decoder = str(DECODER).encode("utf-8")
    config.model_config.tokens = str(TOKENS).encode("utf-8")
    config.model_config.num_threads = 2
    config.model_config.provider = b"cpu"
    config.model_config.debug = 0
    config.decoding_method = b"greedy_search"
    config.max_active_paths = 4
    config.enable_endpoint = 0
    recognizer = lib.SherpaOnnxCreateOnlineRecognizer(ctypes.byref(config))
    if not recognizer:
        print("ERROR: sherpa-onnx recognizer 创建失败")
        sys.exit(1)
    return recognizer


def sherpa_decode(lib, recognizer, samples, sr):
    """sherpa-onnx fork 流式解码（带 is_final）"""
    stream = lib.SherpaOnnxCreateOnlineStream(recognizer)
    if not stream:
        return ""

    n_samples = len(samples)
    c_samples = (ctypes.c_float * n_samples)(*samples)

    chunk_size = int(0.6 * sr)
    offset = 0
    while offset < n_samples:
        end = min(offset + chunk_size, n_samples)
        chunk_len = end - offset
        chunk_ptr = ctypes.cast(
            ctypes.byref(c_samples, offset * ctypes.sizeof(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_float))
        lib.SherpaOnnxOnlineStreamAcceptWaveform(stream, sr, chunk_ptr, chunk_len)
        while lib.SherpaOnnxIsOnlineStreamReady(recognizer, stream):
            lib.SherpaOnnxDecodeOnlineStream(recognizer, stream)
        offset = end

    # SetOption("is_final", "true") + tail silence + InputFinished
    lib.SherpaOnnxOnlineStreamSetOption(stream, b"is_final", b"true")
    tail_len = int(0.3 * sr)
    tail = (ctypes.c_float * tail_len)(*([0.0] * tail_len))
    lib.SherpaOnnxOnlineStreamAcceptWaveform(stream, sr, tail, tail_len)
    lib.SherpaOnnxOnlineStreamInputFinished(stream)
    while lib.SherpaOnnxIsOnlineStreamReady(recognizer, stream):
        lib.SherpaOnnxDecodeOnlineStream(recognizer, stream)

    result_ptr = lib.SherpaOnnxGetOnlineStreamResult(recognizer, stream)
    text = ""
    if result_ptr and result_ptr.contents.text:
        text = result_ptr.contents.text.decode("utf-8", errors="replace").strip()
    if result_ptr:
        lib.SherpaOnnxDestroyOnlineRecognizerResult(result_ptr)
    lib.SherpaOnnxDestroyOnlineStream(stream)
    return text


# ──────────────────────────────────────────────────
# FunASR 流式 Paraformer
# ──────────────────────────────────────────────────
def create_funasr_model():
    from funasr import AutoModel
    model = AutoModel(
        model="paraformer-zh-streaming",
        disable_update=True,
    )
    return model


def funasr_decode(model, audio_path):
    """FunASR 流式解码（整文件传入，内部自动分 chunk + is_final=True）"""
    cache = {}
    res = model.generate(
        input=audio_path,
        cache=cache,
        is_final=True,
        chunk_size=[0, 10, 5],  # left=0, center=10 (0.6s), right=5
    )
    if res and len(res) > 0:
        return res[0].get("text", "").strip()
    return ""


# ──────────────────────────────────────────────────
# WAV 读取 / CER
# ──────────────────────────────────────────────────
def read_wav(path):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        sr = f.getframerate()
        n = f.getnframes()
        data = f.readframes(n)
    samples = struct.unpack(f"<{len(data)//2}h", data)
    return [s / 32768.0 for s in samples], sr


def levenshtein(a, b):
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def normalize_text(text):
    text = re.sub(r'[，。！？、；：\u201c\u201d\u2018\u2019（）【】《》…—\u2013·,.!?;:\'"()\[\]{}<>]', '', text)
    text = text.replace(" ", "").lower()
    return text


def compute_cer(actual, expected):
    if isinstance(expected, list):
        return min(compute_cer(actual, e) for e in expected)
    a = normalize_text(actual)
    e = normalize_text(expected)
    if not e:
        return 0.0 if not a else 1.0
    return levenshtein(list(a), list(e)) / len(list(e))


# ──────────────────────────────────────────────────
# 语料加载
# ──────────────────────────────────────────────────
def load_all_entries():
    entries = []
    if CORPUS_JSON.exists():
        with open(CORPUS_JSON) as f:
            entries.extend(json.load(f).get("entries", []))
    if REAL_MANIFEST.exists():
        with open(REAL_MANIFEST) as f:
            entries.extend(json.load(f).get("entries", []))
    return entries


def resolve_audio_path(entry):
    audio_files = entry.get("audio_files", {})
    for key in ["real", "edge_tts", "synthetic"]:
        if key in audio_files:
            return str(FIXTURES / audio_files[key])
    if audio_files:
        return str(FIXTURES / list(audio_files.values())[0])
    return None


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────
def main():
    for p in [ENCODER, DECODER, TOKENS]:
        if not p.exists():
            print(f"ERROR: 模型不存在: {p}")
            sys.exit(1)
    if not SHERPA_LIB.exists():
        print(f"ERROR: 库不存在: {SHERPA_LIB}")
        sys.exit(1)

    print("=" * 80)
    print("FunASR vs sherpa-onnx fork (is_final) Benchmark")
    print("=" * 80)
    print(f"sherpa-onnx: {SHERPA_LIB.name} (INT8 ONNX)")
    print(f"FunASR:      paraformer-zh-streaming (PyTorch FP32)")
    print()

    # 加载 sherpa-onnx
    print("加载 sherpa-onnx...")
    lib = load_sherpa_lib()
    recognizer = create_sherpa_recognizer(lib)

    # 加载 FunASR
    print("加载 FunASR...")
    funasr_model = create_funasr_model()

    entries = load_all_entries()
    print(f"\n加载 {len(entries)} 条语料\n")

    cer_sherpa_list = []
    cer_funasr_list = []
    total = 0

    fmt = "  [{i:2d}/{n}] {eid:30s}  sherpa={cs:.3f}  funasr={cf:.3f}  {marker}"

    for i, entry in enumerate(entries):
        audio_path = resolve_audio_path(entry)
        if not audio_path or not os.path.exists(audio_path):
            print(f"  [{i+1:2d}/{len(entries)}] {entry.get('id','?'):30s}  音频不存在，跳过")
            continue

        samples, sr = read_wav(audio_path)
        expected = entry.get("expected_text", "")

        # sherpa-onnx fork (is_final)
        text_sherpa = sherpa_decode(lib, recognizer, samples, sr)

        # FunASR
        text_funasr = funasr_decode(funasr_model, audio_path)

        cer_sherpa = compute_cer(text_sherpa, expected)
        cer_funasr = compute_cer(text_funasr, expected)

        total += 1
        cer_sherpa_list.append(cer_sherpa)
        cer_funasr_list.append(cer_funasr)

        marker = ""
        diff = cer_sherpa - cer_funasr
        if abs(diff) > 0.01:
            if diff < 0:
                marker = "sherpa better"
            else:
                marker = "funasr better"

        eid = entry.get("id", f"entry_{i}")
        print(fmt.format(i=i+1, n=len(entries), eid=eid, cs=cer_sherpa, cf=cer_funasr, marker=marker))

        if text_sherpa != text_funasr:
            print(f"           sherpa: \"{text_sherpa}\"")
            print(f"           funasr: \"{text_funasr}\"")

    # 汇总
    if cer_sherpa_list:
        avg_sherpa = sum(cer_sherpa_list) / len(cer_sherpa_list)
        avg_funasr = sum(cer_funasr_list) / len(cer_funasr_list)
        sherpa_wins = sum(1 for s, f in zip(cer_sherpa_list, cer_funasr_list) if s < f - 0.01)
        funasr_wins = sum(1 for s, f in zip(cer_sherpa_list, cer_funasr_list) if f < s - 0.01)
        ties = total - sherpa_wins - funasr_wins

        print()
        print("═" * 80)
        print(f"  总计: {total} 条")
        print(f"  平均 CER (sherpa-onnx fork): {avg_sherpa:.4f}")
        print(f"  平均 CER (FunASR):           {avg_funasr:.4f}")
        print(f"  sherpa 胜: {sherpa_wins}  平局: {ties}  FunASR 胜: {funasr_wins}")
        print("═" * 80)

    lib.SherpaOnnxDestroyOnlineRecognizer(recognizer)


if __name__ == "__main__":
    main()
