#!/usr/bin/env python3
"""
验证 sherpa-onnx fork 的 is_final 功能。

纯 ctypes 方式调用我们编译的 C API 库（不依赖 sherpa_onnx Python binding）。

对语料分别跑两遍流式 Paraformer：
  - 对照组：不调用 SetOption（原始行为）
  - 实验组：在最后一个 chunk 前调用 SetOption("is_final", "true")

对比：
  1. 中间 chunk 输出完全一致
  2. 实验组尾部 token 更完整（CER 更低）
"""

import ctypes
import json
import os
import struct
import sys
import wave
from pathlib import Path

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
# C API ctypes 绑定
# ──────────────────────────────────────────────────

# 结构体定义（需要与 c-api.h 中的结构体对应）
class SherpaOnnxOnlineTransducerModelConfig(ctypes.Structure):
    _fields_ = [
        ("encoder", ctypes.c_char_p),
        ("decoder", ctypes.c_char_p),
        ("joiner", ctypes.c_char_p),
    ]

class SherpaOnnxOnlineParaformerModelConfig(ctypes.Structure):
    _fields_ = [
        ("encoder", ctypes.c_char_p),
        ("decoder", ctypes.c_char_p),
    ]

class SherpaOnnxOnlineZipformer2CtcModelConfig(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_char_p),
    ]

class SherpaOnnxOnlineNemoCtcModelConfig(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_char_p),
    ]

class SherpaOnnxOnlineToneCtcModelConfig(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_char_p),
    ]

class SherpaOnnxOnlineModelConfig(ctypes.Structure):
    _fields_ = [
        ("transducer", SherpaOnnxOnlineTransducerModelConfig),
        ("paraformer", SherpaOnnxOnlineParaformerModelConfig),
        ("zipformer2_ctc", SherpaOnnxOnlineZipformer2CtcModelConfig),
        ("tokens", ctypes.c_char_p),
        ("num_threads", ctypes.c_int32),
        ("provider", ctypes.c_char_p),
        ("debug", ctypes.c_int32),
        ("model_type", ctypes.c_char_p),
        ("modeling_unit", ctypes.c_char_p),
        ("bpe_vocab", ctypes.c_char_p),
        ("tokens_buf", ctypes.c_char_p),
        ("tokens_buf_size", ctypes.c_int32),
        ("nemo_ctc", SherpaOnnxOnlineNemoCtcModelConfig),
        ("t_one_ctc", SherpaOnnxOnlineToneCtcModelConfig),
    ]

class SherpaOnnxFeatureConfig(ctypes.Structure):
    _fields_ = [
        ("sample_rate", ctypes.c_int32),
        ("feature_dim", ctypes.c_int32),
    ]

class SherpaOnnxOnlineCtcFstDecoderConfig(ctypes.Structure):
    _fields_ = [
        ("graph", ctypes.c_char_p),
        ("max_active", ctypes.c_int32),
    ]

class SherpaOnnxHomophoneReplacerConfig(ctypes.Structure):
    _fields_ = [
        ("dict_dir", ctypes.c_char_p),
        ("lexicon", ctypes.c_char_p),
        ("rule_fsts", ctypes.c_char_p),
    ]

class SherpaOnnxOnlineRecognizerConfig(ctypes.Structure):
    _fields_ = [
        ("feat_config", SherpaOnnxFeatureConfig),
        ("model_config", SherpaOnnxOnlineModelConfig),
        ("decoding_method", ctypes.c_char_p),
        ("max_active_paths", ctypes.c_int32),
        ("enable_endpoint", ctypes.c_int32),
        ("rule1_min_trailing_silence", ctypes.c_float),
        ("rule2_min_trailing_silence", ctypes.c_float),
        ("rule3_min_utterance_length", ctypes.c_float),
        ("hotwords_file", ctypes.c_char_p),
        ("hotwords_score", ctypes.c_float),
        ("ctc_fst_decoder_config", SherpaOnnxOnlineCtcFstDecoderConfig),
        ("rule_fsts", ctypes.c_char_p),
        ("rule_fars", ctypes.c_char_p),
        ("blank_penalty", ctypes.c_float),
        ("hotwords_buf", ctypes.c_char_p),
        ("hotwords_buf_size", ctypes.c_int32),
        ("hr", SherpaOnnxHomophoneReplacerConfig),
    ]

class SherpaOnnxOnlineRecognizerResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("tokens", ctypes.c_char_p),
        ("tokens_arr", ctypes.POINTER(ctypes.c_char_p)),
        ("timestamps", ctypes.POINTER(ctypes.c_float)),
        ("count", ctypes.c_int32),
        ("json", ctypes.c_char_p),
    ]


def load_lib():
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
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
    ]
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

    lib.SherpaOnnxOnlineStreamReset.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.SherpaOnnxOnlineStreamReset.restype = None

    return lib


def create_recognizer(lib):
    config = SherpaOnnxOnlineRecognizerConfig()
    # 清零整个结构体
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
        print("ERROR: 创建 OnlineRecognizer 失败")
        sys.exit(1)
    return recognizer


# ──────────────────────────────────────────────────
# WAV 读取
# ──────────────────────────────────────────────────
def read_wav(path: str) -> tuple:
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1, f"Expected mono, got {f.getnchannels()} channels"
        assert f.getsampwidth() == 2, f"Expected 16-bit, got {f.getsampwidth()*8}-bit"
        sr = f.getframerate()
        n = f.getnframes()
        data = f.readframes(n)
    samples = struct.unpack(f"<{len(data)//2}h", data)
    return [s / 32768.0 for s in samples], sr


# ──────────────────────────────────────────────────
# CER 计算
# ──────────────────────────────────────────────────
def levenshtein(a, b):
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


import re
def normalize_text(text: str) -> str:
    text = re.sub(r'[，。！？、；：""''（）【】《》…—–·,.!?;:\'"()\[\]{}<>]', '', text)
    text = text.replace(" ", "").lower()
    return text


def compute_cer(actual: str, expected) -> float:
    if isinstance(expected, list):
        return min(compute_cer(actual, e) for e in expected)
    a = normalize_text(actual)
    e = normalize_text(expected)
    if not e:
        return 0.0 if not a else 1.0
    return levenshtein(list(a), list(e)) / len(list(e))


# ──────────────────────────────────────────────────
# 流式解码（纯 ctypes C API）
# ──────────────────────────────────────────────────
def decode_streaming(lib, recognizer, samples, sr, use_final_chunk=False):
    """
    流式解码一条音频，返回 (最终识别文本, [中间结果列表])
    中间结果列表用于对比两组中间 chunk 是否一致。
    """
    stream = lib.SherpaOnnxCreateOnlineStream(recognizer)
    if not stream:
        return "", []

    # 转为 ctypes float 数组
    n_samples = len(samples)
    c_samples = (ctypes.c_float * n_samples)(*samples)

    # 分 chunk 送入（模拟流式，每次 0.6s）
    chunk_size = int(0.6 * sr)
    offset = 0
    intermediate_results = []

    while offset < n_samples:
        end = min(offset + chunk_size, n_samples)
        chunk_len = end - offset
        chunk_ptr = ctypes.cast(ctypes.byref(c_samples, offset * ctypes.sizeof(ctypes.c_float)),
                                ctypes.POINTER(ctypes.c_float))
        lib.SherpaOnnxOnlineStreamAcceptWaveform(stream, sr, chunk_ptr, chunk_len)

        while lib.SherpaOnnxIsOnlineStreamReady(recognizer, stream):
            lib.SherpaOnnxDecodeOnlineStream(recognizer, stream)

        # 记录中间结果
        result_ptr = lib.SherpaOnnxGetOnlineStreamResult(recognizer, stream)
        if result_ptr and result_ptr.contents.text:
            text = result_ptr.contents.text.decode("utf-8", errors="replace").strip()
            intermediate_results.append(text)
        else:
            intermediate_results.append("")
        if result_ptr:
            lib.SherpaOnnxDestroyOnlineRecognizerResult(result_ptr)

        offset = end

    # 最后处理
    if use_final_chunk:
        lib.SherpaOnnxOnlineStreamSetOption(stream, b"is_final", b"true")

    # 添加尾部静音并标记结束
    tail_len = int(0.3 * sr)
    tail = (ctypes.c_float * tail_len)(*([0.0] * tail_len))
    lib.SherpaOnnxOnlineStreamAcceptWaveform(stream, sr, tail, tail_len)
    lib.SherpaOnnxOnlineStreamInputFinished(stream)

    while lib.SherpaOnnxIsOnlineStreamReady(recognizer, stream):
        lib.SherpaOnnxDecodeOnlineStream(recognizer, stream)

    # 获取最终结果
    result_ptr = lib.SherpaOnnxGetOnlineStreamResult(recognizer, stream)
    final_text = ""
    if result_ptr and result_ptr.contents.text:
        final_text = result_ptr.contents.text.decode("utf-8", errors="replace").strip()
    if result_ptr:
        lib.SherpaOnnxDestroyOnlineRecognizerResult(result_ptr)

    lib.SherpaOnnxDestroyOnlineStream(stream)
    return final_text, intermediate_results


# ──────────────────────────────────────────────────
# 语料加载
# ──────────────────────────────────────────────────
def load_all_entries():
    entries = []
    if CORPUS_JSON.exists():
        with open(CORPUS_JSON) as f:
            data = json.load(f)
            entries.extend(data.get("entries", []))
    if REAL_MANIFEST.exists():
        with open(REAL_MANIFEST) as f:
            data = json.load(f)
            entries.extend(data.get("entries", []))
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
            print(f"ERROR: 模型文件不存在: {p}")
            sys.exit(1)

    if not SHERPA_LIB.exists():
        print(f"ERROR: 共享库不存在: {SHERPA_LIB}")
        sys.exit(1)

    print("=" * 70)
    print("sherpa-onnx is_final 验证测试 (纯 ctypes C API)")
    print("=" * 70)
    print(f"模型:  {MODEL_DIR.name}")
    print(f"库:    {SHERPA_LIB}")
    print()

    lib = load_lib()
    recognizer = create_recognizer(lib)

    entries = load_all_entries()
    print(f"加载 {len(entries)} 条语料")
    print()

    results_baseline = []
    results_is_final = []
    mid_chunk_diffs = 0
    total_tested = 0

    for i, entry in enumerate(entries):
        audio_path = resolve_audio_path(entry)
        if not audio_path or not os.path.exists(audio_path):
            print(f"  [{i+1:2d}/{len(entries)}] {entry.get('id','?'):30s}  音频不存在，跳过")
            continue

        samples, sr = read_wav(audio_path)
        expected = entry.get("expected_text", "")

        # 对照组
        text_baseline, mid_baseline = decode_streaming(lib, recognizer, samples, sr, use_final_chunk=False)
        # 实验组
        text_is_final, mid_is_final = decode_streaming(lib, recognizer, samples, sr, use_final_chunk=True)

        total_tested += 1

        # 检查中间 chunk 是否一致（排除最后一个，因为最后一个可能不同）
        min_mid = min(len(mid_baseline), len(mid_is_final))
        mid_match = True
        if min_mid > 1:
            # 比较除最后一个外的所有中间结果
            for j in range(min_mid - 1):
                if mid_baseline[j] != mid_is_final[j]:
                    mid_match = False
                    break
        if not mid_match:
            mid_chunk_diffs += 1

        cer_baseline = compute_cer(text_baseline, expected)
        cer_is_final = compute_cer(text_is_final, expected)

        marker = ""
        if cer_is_final < cer_baseline - 0.01:
            marker = " ✓ improved"
        elif cer_is_final > cer_baseline + 0.01:
            marker = " ✗ regressed"

        mid_marker = "  mid:OK" if mid_match else "  mid:DIFF!"

        entry_id = entry.get("id", f"entry_{i}")
        print(f"  [{i+1:2d}/{len(entries)}] {entry_id:30s}  "
              f"baseline={cer_baseline:.3f}  is_final={cer_is_final:.3f}{marker}{mid_marker}")

        if text_baseline != text_is_final:
            print(f"           baseline: \"{text_baseline}\"")
            print(f"           is_final: \"{text_is_final}\"")

        results_baseline.append(cer_baseline)
        results_is_final.append(cer_is_final)

    # 汇总
    if results_baseline:
        avg_baseline = sum(results_baseline) / len(results_baseline)
        avg_is_final = sum(results_is_final) / len(results_is_final)
        improved = sum(1 for b, f in zip(results_baseline, results_is_final) if f < b - 0.01)
        regressed = sum(1 for b, f in zip(results_baseline, results_is_final) if f > b + 0.01)
        same = len(results_baseline) - improved - regressed

        print()
        print("═" * 70)
        print(f"  总计: {total_tested} 条")
        print(f"  平均 CER (baseline): {avg_baseline:.4f}")
        print(f"  平均 CER (is_final): {avg_is_final:.4f}")
        print(f"  改善: {improved}  不变: {same}  退化: {regressed}")
        print(f"  中间 chunk 不一致: {mid_chunk_diffs} 条")
        print("═" * 70)

    lib.SherpaOnnxDestroyOnlineRecognizer(recognizer)


if __name__ == "__main__":
    main()
