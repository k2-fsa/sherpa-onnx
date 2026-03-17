#!/usr/bin/env python3

# Real-time keyword spotting from a microphone with sherpa-onnx Python API
#
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
# to download pre-trained models

import argparse
import sys
from pathlib import Path

from typing import List

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_onnx

# 默认模型目录，可按需修改
DEFAULT_MODEL_DIR = Path("/Users/cece/Documents/kws/sherpa/sherpa-onnx-model")


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html to download it"
    )


def load_keyword_token_sets(keywords_file: str) -> dict:
    """
    从 keywords 文件解析：每个关键词（@ 后面的显示名）对应的 token 序列。
    返回 (token_tuple -> display_keyword)，用于「整段解码 full_tokens 完全等于某关键词」才采纳。
    """
    # tuple(tokens) -> display keyword，例如 ("g", "uò", "l", "ái") -> "过来"
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default=str(DEFAULT_MODEL_DIR / "tokens.txt"),
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default=str(DEFAULT_MODEL_DIR / "encoder-epoch-13-avg-2-chunk-16-left-64.onnx"),
        help="Path to the transducer encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        default=str(DEFAULT_MODEL_DIR / "decoder-epoch-13-avg-2-chunk-16-left-64.onnx"),
        help="Path to the transducer decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        default=str(DEFAULT_MODEL_DIR / "joiner-epoch-13-avg-2-chunk-16-left-64.onnx"),
        help="Path to the transducer joiner model",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--max-active-paths",
        type=int,
        default=4,
        help="""
        It specifies number of active paths to keep during decoding.
        """,
    )

    parser.add_argument(
        "--num-trailing-blanks",
        type=int,
        default=3,
        help="""The number of trailing blanks a keyword should be followed.
        Keep low (1~2) so single keywords trigger; use full_tokens exact-match to
        reject phrases like 别过来 triggering 过来.
        """,
    )

    parser.add_argument(
        "--keywords-file",
        type=str,
        default=str(DEFAULT_MODEL_DIR / "keywords.txt"),
        help="""
        The file containing keywords, one words/phrases per line, and for each
        phrase the bpe/cjkchar/pinyin are separated by a space. For example:

        ▁HE LL O ▁WORLD
        x iǎo ài t óng x ué 
        """,
    )

    parser.add_argument(
        "--keywords-score",
        type=float,
        default=2.0,
        help="""
        The boosting score of each token for keywords. The larger the easier to
        survive beam search.
        """,
    )

    parser.add_argument(
        "--keywords-threshold",
        type=float,
        default=0.15,
        help="""
        The trigger threshold (i.e. probability) of the keyword. The larger the
        harder to trigger.
        """,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print when engine triggers but full_tokens does not match (for debugging).",
    )

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

    assert Path(
        args.keywords_file
    ).is_file(), (
        f"keywords_file : {args.keywords_file} not exist, please provide a valid path."
    )

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

    # 整段 full_tokens 与某关键词完全一致才采纳（说「不要过来」不触发「过来」）
    token_tuple_to_keyword = load_keyword_token_sets(args.keywords_file)

    print("Started! Please speak (说「别动」或「过来」测试)")
    print("仅当整段发音完全等于某关键词时才命中（说「不要过来」不触发「过来」）")
    print("-" * 40)

    idx = 0
    block_count = 0

    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    stream = keyword_spotter.create_stream()
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)
            while keyword_spotter.is_ready(stream):
                keyword_spotter.decode_stream(stream)
                # 用底层 C 扩展的 get_result，返回带 .keyword 和 .full_tokens 的完整结果
                res = keyword_spotter.keyword_spotter.get_result(stream)
                if res.keyword.strip():
                    full_tokens = getattr(res, "full_tokens", None)
                    if full_tokens is not None:
                        # 有 full_tokens：只有整段解码完全等于某关键词才采纳
                        full_tuple = tuple(full_tokens)
                        matched_keyword = token_tuple_to_keyword.get(full_tuple)
                        if matched_keyword is not None:
                            print(f"[命中] #{idx}: {matched_keyword}")
                            idx += 1
                        elif args.debug:
                            print(
                                f"[debug] 触发 keyword={res.keyword!r} 但 full_tokens 不匹配: {list(full_tuple)}"
                            )
                    else:
                        # 旧版 C 扩展无 full_tokens，退化为直接采纳引擎命中的 keyword
                        print(f"[命中] #{idx}: {res.keyword.strip()}")
                        idx += 1
                    keyword_spotter.reset_stream(stream)
            block_count += 1
            if block_count % 50 == 0:
                print(".", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
