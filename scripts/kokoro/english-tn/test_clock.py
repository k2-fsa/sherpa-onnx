#!/usr/bin/env python3
# Copyright (c) 2026 Hedy AI
"""Compare compiled time rules with NeMo's deterministic tagger/verbalizer."""
import argparse
import json
import pathlib
import sys

import kaldifst
from pynini.lib import rewrite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-source", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.nemo_source))

    from nemo_text_processing.text_normalization.en.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.en.taggers.time import TimeFst as Tagger
    from nemo_text_processing.text_normalization.en.verbalizers.time import TimeFst as Verbalizer

    cardinal = CardinalFst(deterministic=True)
    clock = (
        Tagger(cardinal, deterministic=True).fst
        @ Verbalizer(deterministic=True).fst
    ).optimize()
    numeric = kaldifst.TextNormalizer(str(args.bundle / "english_tn.fst"))
    symbols = kaldifst.TextNormalizer(str(args.bundle / "english_symbols.fst"))

    texts = set()
    for hour in range(24):
        for minute in [0, 1, 5, 9, 10, 15, 20, 23, 30, 45, 59]:
            for hour_text in {str(hour), f"{hour:02d}"}:
                suffixes = [""]
                if 1 <= hour <= 12:
                    suffixes += [" AM", " PM", " am", " pm", "AM", "PM", "am", "pm"]
                texts.update(
                    f"{hour_text}:{minute:02d}" + suffix for suffix in suffixes
                )
    for hour in [1, 9, 12, 23]:
        texts.update([f"{hour}:00:05", f"{hour}:15:00"])

    rows = []
    for text in sorted(texts):
        expected = symbols.normalize(rewrite.one_top_rewrite(text, clock))
        output = symbols.normalize(numeric.normalize(text))
        rows.append(dict(text=text, expected=expected, output=output, exact=output == expected))
    for text, expected in [
        ("Time:9:00", "Time:nine o'clock"),
        (
            "I am available at 9:00 am and the pm is free.",
            "I am available at nine a m and the pm is free.",
        ),
    ]:
        output = symbols.normalize(numeric.normalize(text))
        rows.append(dict(text=text, expected=expected, output=output, exact=output == expected))

    result = dict(
        cases=len(rows),
        mismatches=sum(not row["exact"] for row in rows),
        reference="Pinned NeMo deterministic time tagger/verbalizer plus symbol FST",
        rows=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))
    assert result["mismatches"] == 0, "Compiled rules differ from NeMo time output"


if __name__ == "__main__":
    main()
