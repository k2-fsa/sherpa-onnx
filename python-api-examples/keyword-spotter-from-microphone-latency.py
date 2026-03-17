#!/usr/bin/env python3
"""
KWS 延迟统计脚本：在 keyword-spotter-from-microphone 逻辑基础上，统计
「用户说完话 → 模型输出结果」的延迟。

约定：以「发送最后一个音频块」的时刻近似为用户说完话的时刻；
     以 get_result() 返回命中为模型输出结果的时刻。
延迟 = 模型输出时刻 - 本轮发送该音频块的时刻（单位 ms）。

用法与 keyword-spotter-from-microphone.py 相同，仅增加延迟打印与汇总。
Ctrl+C 退出时打印本次会话的延迟统计（次数、平均/最小/最大 ms）。
"""

import argparse
import sys
import time
from pathlib import Path

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    sys.exit(-1)

import sherpa_onnx

DEFAULT_MODEL_DIR = Path("/Users/cece/Documents/kws/sherpa/sherpa-onnx-model")


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html to download it"
    )


def load_keyword_token_sets(keywords_file: str) -> dict:
    token_tuple_to_keyword = {}
    with open(keywords_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            tokens = []
            display = ""
            for w in parts:
                if w.startswith("@"):
                    display = w[1:].strip()
                    break
                tokens.append(w)
            if display and tokens:
                token_tuple_to_keyword[tuple(tokens)] = display
    return token_tuple_to_keyword


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="KWS 麦克风实时检测 + 延迟统计",
    )
    parser.add_argument("--tokens", type=str, default=str(DEFAULT_MODEL_DIR / "tokens.txt"))
    parser.add_argument(
        "--encoder",
        type=str,
        default=str(DEFAULT_MODEL_DIR / "encoder-epoch-13-avg-2-chunk-16-left-64.onnx"),
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default=str(DEFAULT_MODEL_DIR / "decoder-epoch-13-avg-2-chunk-16-left-64.onnx"),
    )
    parser.add_argument(
        "--joiner",
        type=str,
        default=str(DEFAULT_MODEL_DIR / "joiner-epoch-13-avg-2-chunk-16-left-64.onnx"),
    )
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--provider", type=str, default="cpu")
    parser.add_argument("--max-active-paths", type=int, default=4)
    parser.add_argument("--num-trailing-blanks", type=int, default=1)
    parser.add_argument("--keywords-file", type=str, default=str(DEFAULT_MODEL_DIR / "keywords.txt"))
    parser.add_argument("--keywords-score", type=float, default=2.0)
    parser.add_argument("--keywords-threshold", type=float, default=0.15)
    parser.add_argument("--debug", action="store_true", help="触发但 full_tokens 不匹配时打印")
    return parser.parse_args()


def main():
    args = get_args()

    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)
    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    assert_file_exists(args.tokens)
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert Path(args.keywords_file).is_file(), f"keywords_file not exist: {args.keywords_file}"

    keyword_spotter = sherpa_onnx.KeywordSpotter(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=args.num_threads,
        max_active_paths=args.max_active_paths,
        keywords_file=args.keywords_file,
        keywords_score=args.keywords_score,
        keywords_threshold=args.keywords_threshold,
        num_trailing_blanks=args.num_trailing_blanks,
        provider=args.provider,
    )

    token_tuple_to_keyword = load_keyword_token_sets(args.keywords_file)

    # 延迟统计：每次命中的延迟（ms），模块级以便 Ctrl+C 时打印汇总
    global latencies_ms
    latencies_ms = []

    print("Started! 说关键词测试，将统计「说完 → 出结果」延迟（ms）")
    print("约定：以发送最后一个音频块时刻为说完时刻")
    print("-" * 50)

    idx = 0
    block_count = 0
    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)
    stream = keyword_spotter.create_stream()

    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)
            # 以「本块已送入」的时刻近似为「用户说完话」的截止时刻
            t_after_chunk = time.perf_counter()

            while keyword_spotter.is_ready(stream):
                keyword_spotter.decode_stream(stream)
                res = keyword_spotter.keyword_spotter.get_result(stream)
                if res.keyword.strip():
                    t_result = time.perf_counter()
                    latency_ms = (t_result - t_after_chunk) * 1000.0
                    full_tokens = getattr(res, "full_tokens", None)
                    if full_tokens is not None:
                        full_tuple = tuple(full_tokens)
                        matched_keyword = token_tuple_to_keyword.get(full_tuple)
                        if matched_keyword is not None:
                            latencies_ms.append(latency_ms)
                            print(f"[命中] #{idx}: {matched_keyword}  延迟: {latency_ms:.1f} ms")
                            idx += 1
                        elif args.debug:
                            print(f"[debug] 触发 keyword={res.keyword!r} 但 full_tokens 不匹配: {list(full_tuple)}")
                    else:
                        latencies_ms.append(latency_ms)
                        print(f"[命中] #{idx}: {res.keyword.strip()}  延迟: {latency_ms:.1f} ms")
                        idx += 1
                    keyword_spotter.reset_stream(stream)
            block_count += 1
            if block_count % 50 == 0:
                print(".", flush=True)


# 供 Ctrl+C 时打印汇总
latencies_ms = []

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
        if latencies_ms:
            n = len(latencies_ms)
            avg_ms = sum(latencies_ms) / n
            min_ms = min(latencies_ms)
            max_ms = max(latencies_ms)
            print("\n" + "=" * 50)
            print("延迟统计（说完 → 模型输出）")
            print(f"  命中次数: {n}")
            print(f"  平均延迟: {avg_ms:.1f} ms")
            print(f"  最小延迟: {min_ms:.1f} ms")
            print(f"  最大延迟: {max_ms:.1f} ms")
            print("=" * 50)
        else:
            print("\n本次未命中关键词，无延迟数据。")
