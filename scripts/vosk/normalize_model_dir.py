#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/normalize_model_dir.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Rearrange a vosk model dir into the standard ("v2") directory layout.
#
# Some models (e.g. vosk-model-small-de-zamia-0.3,
# vosk-model-small-fr-pguyot-0.3) ship the old ("v1") FLAT layout --
# final.mdl, mfcc.conf, HCLr.fst, Gr.fst, ... at the top level and no
# conf/ directory at all.  vosk-api's src/model.cc handles both
# (Model::ConfigureV1 / ConfigureV2); the tools here expect the standard
# layout, so this script MOVES the files into place:
#
#   final.mdl              -> am/final.mdl
#   global_cmvn.stats      -> am/global_cmvn.stats
#   mfcc/fbank/pitch.conf  -> conf/
#   model.conf             -> conf/model.conf
#   HCLG/HCLr/Gr.fst, disambig_tid.int, words.txt -> graph/
#   word_boundary.int      -> graph/phones/word_boundary.int
#
# v1 models have no model.conf (ConfigureV1 hardcodes its defaults), so
# one is written with EXACTLY those defaults -- the decode-relevant ones
# anyway -- to keep the recognition identical after the rearrange:
#
#   --max-active=7000 --beam=13.0 --lattice-beam=6.0
#   --acoustic-scale=1.0 --frame-subsampling-factor=3
#
# Already-standard dirs are left alone (except for a missing
# conf/model.conf, which gets the same defaults).  Idempotent.

import argparse
from pathlib import Path

# EXACTLY the vosk-api Model::ConfigureV1 defaults (all of it -- the
# endpoint options matter too: without --endpoint.silence-phones the
# vosk runtime aborts when it loads the model as v2, and the rules keep
# its endpointing identical to the flat-layout behavior)
V1_MODEL_CONF = """\
--max-active=7000
--beam=13.0
--lattice-beam=6.0
--acoustic-scale=1.0
--frame-subsampling-factor=3
--endpoint.silence-phones=1:2:3:4:5:6:7:8:9:10
--endpoint.rule2.min-trailing-silence=0.5
--endpoint.rule3.min-trailing-silence=1.0
--endpoint.rule4.min-trailing-silence=2.0
--print-args=false
"""

AM_FILES = ["final.mdl", "global_cmvn.stats"]
CONF_FILES = ["mfcc.conf", "fbank.conf", "pitch.conf", "model.conf"]
GRAPH_FILES = [
    "HCLG.fst",
    "HCLr.fst",
    "Gr.fst",
    "disambig_tid.int",
    "words.txt",
]
GRAPH_PHONES_FILES = ["word_boundary.int"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        help="a vosk model dir in the standard or the old flat layout",
    )
    return parser.parse_args()


def move(src: Path, dst: Path, root: Path):
    print(f"  {src.relative_to(root)} -> {dst.relative_to(root)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


def main():
    args = get_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"{model_dir} is not a directory")

    flat = (model_dir / "final.mdl").is_file()
    standard = (model_dir / "am" / "final.mdl").is_file()
    if not flat and not standard:
        raise SystemExit(f"{model_dir}: no final.mdl found -- unknown layout")

    if flat and not standard:
        print(f"{model_dir}: rearranging the flat (v1) layout into the standard one")
        for name in AM_FILES:
            if (model_dir / name).is_file():
                move(model_dir / name, model_dir / "am" / name, model_dir)
        for name in CONF_FILES:
            if (model_dir / name).is_file():
                move(model_dir / name, model_dir / "conf" / name, model_dir)
        for name in GRAPH_FILES:
            if (model_dir / name).is_file():
                move(model_dir / name, model_dir / "graph" / name, model_dir)
        for name in GRAPH_PHONES_FILES:
            if (model_dir / name).is_file():
                move(
                    model_dir / name,
                    model_dir / "graph" / "phones" / name,
                    model_dir,
                )

    conf = model_dir / "conf" / "model.conf"
    if not conf.is_file():
        print(f"{model_dir}: writing conf/model.conf with the vosk v1 defaults")
        conf.parent.mkdir(parents=True, exist_ok=True)
        conf.write_text(V1_MODEL_CONF)

    print(f"{model_dir}: standard layout OK")


if __name__ == "__main__":
    main()
