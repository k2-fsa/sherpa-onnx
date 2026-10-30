#!/usr/bin/env python3
# Copyright 2026 Hedy AI. Apache-2.0. Uses NVIDIA NeMo's Apache-2.0 grammars.
import argparse
import json
import pathlib
import sys
import time

parser = argparse.ArgumentParser(
    description="Build a sherpa-compatible English NeMo rule FST"
)
parser.add_argument("--nemo-source", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.nemo_source))
import pynini
from pynini.lib import pynutil, byte, rewrite
from nemo_text_processing.text_normalization.en.taggers.cardinal import (
    CardinalFst as CT,
)
from nemo_text_processing.text_normalization.en.verbalizers.cardinal import (
    CardinalFst as CV,
)
from nemo_text_processing.text_normalization.en.taggers.decimal import DecimalFst as DT
from nemo_text_processing.text_normalization.en.verbalizers.decimal import (
    DecimalFst as DV,
)
from nemo_text_processing.text_normalization.en.taggers.money import MoneyFst as MT
from nemo_text_processing.text_normalization.en.verbalizers.money import MoneyFst as MV
from nemo_text_processing.text_normalization.en.taggers.time import TimeFst as TT
from nemo_text_processing.text_normalization.en.verbalizers.time import TimeFst as TV
from nemo_text_processing.text_normalization.en.taggers.ordinal import OrdinalFst as OT
from nemo_text_processing.text_normalization.en.verbalizers.ordinal import (
    OrdinalFst as OV,
)
from nemo_text_processing.text_normalization.en.taggers.date import (
    get_four_digit_year_graph,
)


def main():
    start = time.perf_counter()
    out = args.output
    out.mkdir(exist_ok=True)
    c = CT(deterministic=True)
    cv = CV(deterministic=True)
    d = DT(c, deterministic=True)
    dv = DV(cv, deterministic=True)
    # Compose classification with verbalization before exporting. Raw NeMo FAR
    # taggers alone produce structured tokens that sherpa does not parse.
    card = (c.fst @ cv.fst).optimize()
    decimal = (d.fst @ dv.fst).optimize()
    money = (
        MT(c, d, deterministic=True).fst @ MV(dv, deterministic=True).fst
    ).optimize()
    # NeMo admits equally weighted dollar/us-dollar synonyms even in deterministic mode.
    sigma_money = byte.BYTE.star
    money = (
        money @ pynini.difference(sigma_money, sigma_money + "us dollar" + sigma_money)
    ).optimize()
    # Direct tagger/verbalizer composition misses whole-currency and .00 forms.
    digits = pynini.closure(byte.DIGIT | ",", 1)
    whole_money = None
    for sym, singular, plural in [
        ("$", "dollar", "dollars"),
        ("£", "pound", "pounds"),
        ("€", "euro", "euros"),
    ]:
        whole = pynini.cross(sym + "1", "one " + singular) | (
            pynutil.delete(sym)
            + (pynini.difference(digits, "1") @ (c.fst @ cv.fst))
            + pynutil.insert(" " + plural)
        )
        whole_money = whole if whole_money is None else whole_money | whole
    whole_money = whole_money.optimize()
    zero_cents = (
        pynini.union("$", "£", "€") + digits + pynutil.delete(".00")
    ) @ whole_money
    # Prefer the complete .00 span over a whole-money prefix ending before the dot.
    money = (money | whole_money | pynutil.add_weight(zero_cents, -0.2)).optimize()
    clock = (TT(c, deterministic=True).fst @ TV(deterministic=True).fst).optimize()
    ordinal = (OT(c, deterministic=True).fst @ OV(deterministic=True).fst).optimize()
    percent = (decimal | card) + pynini.cross("%", " percent")
    year = get_four_digit_year_graph(deterministic=True).optimize()
    sigma = byte.BYTE.star
    # Percent spans take precedence over their numeric substrings.
    # Money and time are rewritten first to prevent competing numeric spans.
    # Boundary restriction intentionally leaves version strings such as v3.14
    # unchanged; this is a measured subset, not a full English TN product.
    graph = pynini.union(
        pynutil.add_weight(percent, -6),
        pynutil.add_weight(decimal, -5),
        pynutil.add_weight(ordinal, -5),
        pynutil.add_weight(card, -1),
    ).optimize()
    alnum = pynini.union(
        *"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ).optimize()
    boundary = pynini.difference(byte.BYTE, alnum).optimize()
    contextual_year = pynini.cdrewrite(
        year,
        pynini.union("In ", "in ", "year ", "Year "),
        pynini.union(boundary, "[EOS]"),
        sigma,
    ).optimize()
    left_boundary = pynini.difference(boundary, ".").optimize()
    numeric = pynini.cdrewrite(
        graph,
        pynini.union(left_boundary, "[BOS]"),
        pynini.union(boundary, "[EOS]"),
        sigma,
    ).optimize()
    monetary = pynini.cdrewrite(
        money,
        pynini.union(left_boundary, "[BOS]"),
        pynini.union(boundary, "[EOS]"),
        sigma,
    ).optimize()
    # In a single weighted numeric union, "12:00 PM" can win as cardinal
    # "12" plus time "00 PM". Separate passes keep complete time fields intact.
    # A colon between digits is not a new time boundary; label colons still are.
    time_left = pynini.union(
        pynini.difference(left_boundary, ":"),
        "[BOS]",
        pynini.difference(byte.BYTE, byte.DIGIT) + ":",
        pynini.accep("[BOS]") + ":",
    )
    time_right = pynini.union(
        pynini.difference(boundary, ":"),
        "[EOS]",
        ":" + pynini.difference(byte.BYTE, byte.DIGIT),
        ":" + pynini.accep("[EOS]"),
    )
    # Consume a meridiem before allowing the bare-clock path to add o'clock.
    # NeMo canonicalizes am/pm only in a time; bare prose "I am" is untouched.
    clock_with_suffix = (
        clock @ (sigma + pynini.union(" AM", " PM") + sigma)
    ).optimize()
    timed_suffix = pynini.cdrewrite(
        clock_with_suffix, time_left, time_right, sigma
    ).optimize()
    timed = pynini.cdrewrite(clock, time_left, time_right, sigma).optimize()
    combined = (contextual_year @ monetary @ timed_suffix @ timed @ numeric).optimize()
    combined.write(str(out / "english_tn.fst"))
    with pynini.Far(
        str(out / "english_tn.far"), mode="w", arc_type="standard", far_type="sttable"
    ) as far:
        far["english_tn"] = combined
    tests = [
        (
            "The budget is 1969 dollars.",
            "The budget is one thousand nine hundred and sixty nine dollars.",
        ),
        (
            "The budget is nineteen sixty nine dollars.",
            "The budget is nineteen sixty nine dollars.",
        ),
        (
            "The price is $12.50 and the increase is 15%.",
            "The price is twelve dollars fifty cents and the increase is fifteen percent.",
        ),
        (
            "In 1969, people watched the moon landing.",
            "In nineteen sixty nine, people watched the moon landing.",
        ),
        ("We meet at 9:30 AM.", "We meet at nine thirty AM."),
        ("12:00 PM", "twelve PM"),
        ("3:00 pm", "three PM"),
        ("9:00", "nine o'clock"),
        ("9:30 am", "nine thirty AM"),
        ("I am calm.", "I am calm."),
        ("There are 42 items.", "There are forty two items."),
        ("The value is -3.14.", "The value is minus three point one four."),
        ("Pay $1.01 or $2.00.", "Pay one dollar one cent or two dollars."),
        ("The increase was 0.5%.", "The increase was zero point five percent."),
        ("This is the 21st item.", "This is the twenty first item."),
        (
            "My neighbors can read aloud.",
            "My neighbors can read aloud.",
        ),
        (
            "The deployment uses SQL, API, and v3.14.",
            "The deployment uses SQL, API, and v3.14.",
        ),
        ("No numbers here: café.", "No numbers here: café."),
    ]
    # These expectations cover english_tn.fst alone; test_runtime.py checks
    # the ordered pack, including the final AM/PM symbol expansion.
    rows = []
    for text, expected in tests:
        t = time.perf_counter()
        actual = rewrite.one_top_rewrite(text, combined)
        ms = (time.perf_counter() - t) * 1000
        rows.append(
            dict(
                input=text,
                expected=expected,
                output=actual,
                exact=actual == expected,
                ms=ms,
            )
        )
        print(rows[-1], flush=True)
    assert all(row["exact"] for row in rows), "Main FST expectation changed"
    result = dict(
        build_s=time.perf_counter() - start,
        states=combined.num_states(),
        fst_bytes=(out / "english_tn.fst").stat().st_size,
        far_bytes=(out / "english_tn.far").stat().st_size,
        rows=rows,
        pynini=pynini.__version__,
        format="OpenFst standard tropical byte labels",
        scope="NeMo composed cardinal/decimal/money/time/ordinal; custom percent wrapper and contextual year rule",
    )
    (out / "build-results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
