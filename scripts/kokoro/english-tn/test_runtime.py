#!/usr/bin/env python3
# Copyright (c) 2026 Hedy AI
"""Measure an ordered sherpa-compatible FST pack on the retained probes."""
import argparse
import json
import pathlib
import kaldifst

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=pathlib.Path, required=True)
parser.add_argument(
    "--cases",
    type=pathlib.Path,
    default=pathlib.Path(__file__).with_name("test_cases.json"),
)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
normalizers = [
    kaldifst.TextNormalizer(str(args.bundle / name))
    for name in ["english_tn.fst", "english_symbols.fst"]
]
rows = []
for case in json.loads(args.cases.read_text(encoding="utf-8")):
    result = case["text"]
    for normalizer in normalizers:
        result = normalizer.normalize(result)
    rows.append(dict(case, output=result, exact=result == case.get("expected")))
args.output.write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(rows, ensure_ascii=True, indent=2))
assert all(
    r["exact"] for r in rows if not r.get("observe")
), "FST regression expectation changed"
