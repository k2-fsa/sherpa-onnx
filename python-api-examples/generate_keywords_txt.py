#!/usr/bin/env python3
"""
从中文关键词列表生成 keywords.txt（sherpa-onnx KWS 所需格式：token1 token2 ... @显示名）。

用法:
  python generate_keywords_txt.py --keywords "静音,别出声,大声" --tokens /path/to/tokens.txt --output keywords.txt
  或
  python generate_keywords_txt.py --keywords-file my_keywords.txt --tokens /path/to/tokens.txt --output keywords.txt

依赖: pip install pypinyin
"""

import argparse
import sys
from pathlib import Path

try:
    from pypinyin import pinyin, Style
    from pypinyin.contrib.tone_convert import to_initials, to_finals_tone
except ImportError:
    print("请先安装 pypinyin:  pip install pypinyin")
    sys.exit(1)


def to_ppinyin(txt: str) -> list:
    """中文 -> 声母+带声调韵母（与 sherpa-onnx ppinyin 一致，tokens.txt 中为 ìng 等）"""
    res = []
    # Style.TONE 得到带声调拼音如 jìng，to_finals_tone 才能得到 ìng
    for x, in pinyin(txt, style=Style.TONE):
        initial = to_initials(x, strict=False)
        final = to_finals_tone(x, strict=False)
        if initial == "" and final == "":
            res.append(x)
        else:
            if initial:
                res.append(initial)
            if final:
                res.append(final)
    return res


def get_args():
    p = argparse.ArgumentParser(description="生成 KWS keywords.txt（ppinyin 格式）")
    p.add_argument(
        "--keywords",
        type=str,
        default="",
        help="逗号分隔的关键词，例如: 静音,别出声,大声",
    )
    p.add_argument(
        "--keywords-file",
        type=str,
        default="",
        help="或提供文件路径，每行一个关键词",
    )
    p.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="模型 tokens.txt 路径",
    )
    p.add_argument(
        "--output",
        type=str,
        default="keywords.txt",
        help="输出 keywords.txt 路径",
    )
    return p.parse_args()


def main():
    args = get_args()

    if args.keywords_file:
        with open(args.keywords_file, "r", encoding="utf-8") as f:
            keywords = [line.strip() for line in f if line.strip()]
    elif args.keywords:
        keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    else:
        print("请指定 --keywords 或 --keywords-file")
        sys.exit(1)

    if not keywords:
        print("关键词列表为空")
        sys.exit(1)

    # 加载 tokens.txt 中合法 token 集合（用于检查并跳过 OOV）
    tokens_set = set()
    with open(args.tokens, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) >= 2:
                tokens_set.add(toks[0])

    out_lines = []
    skipped = []
    for kw in keywords:
        tokens = to_ppinyin(kw)
        oov = [t for t in tokens if t not in tokens_set]
        if oov:
            skipped.append((kw, oov))
            continue
        # 格式: token1 token2 ... @显示名
        out_lines.append(" ".join(tokens) + f" @{kw}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"已写入 {len(out_lines)} 条关键词 -> {args.output}")
    if skipped:
        print(f"跳过 {len(skipped)} 条（含 tokens 表中不存在的音）:")
        for kw, oov in skipped[:10]:
            print(f"  {kw} -> OOV: {oov}")
        if len(skipped) > 10:
            print(f"  ... 共 {len(skipped)} 条")


if __name__ == "__main__":
    main()
