#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="Input directory.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    return parser.parse_args()


def process(out_dir: Path, whl: Path):
    tmp_dir = out_dir / "tmp"
    subprocess.check_call(f"unzip {whl} -d {tmp_dir}", shell=True)
    if "cp37" in str(whl):
        py_version = "3.7"
    elif "cp38" in str(whl):
        py_version = "3.8"
    elif "cp39" in str(whl):
        py_version = "3.9"
    elif "cp310" in str(whl):
        py_version = "3.10"
    elif "cp311" in str(whl):
        py_version = "3.11"
    else:
        py_version = "3.12"

    rpath_list = [
        f"$ORIGIN/../lib/python{py_version}/site-packages/sherpa_onnx/lib",
        f"$ORIGIN/../lib/python{py_version}/dist-packages/sherpa_onnx/lib",
        #
        f"$ORIGIN/../lib/python{py_version}/site-packages/sherpa_onnx/lib64",
        f"$ORIGIN/../lib/python{py_version}/dist-packages/sherpa_onnx/lib64",
        #
        f"$ORIGIN/../lib/python{py_version}/site-packages/sherpa_onnx.libs",
    ]
    rpaths = ":".join(rpath_list)

    for filename in glob.glob(
        f"{tmp_dir}/sherpa_onnx-*data/data/bin/*", recursive=True
    ):
        print(filename)
        existing_rpath = (
            subprocess.check_output(["patchelf", "--print-rpath", filename])
            .decode()
            .strip()
        )
        target_rpaths = rpaths + ":" + existing_rpath
        subprocess.check_call(
            f"patchelf --force-rpath --set-rpath '{target_rpaths}' {filename}",
            shell=True,
        )

    outwheel = Path(shutil.make_archive(whl, "zip", tmp_dir))
    Path(outwheel).rename(out_dir / whl.name)

    shutil.rmtree(tmp_dir)


def main():
    args = get_args()
    print(args)
    in_dir = args.in_dir
    out_dir = args.out_dir
    out_dir.mkdir(exist_ok=True, parents=True)

    for whl in in_dir.glob("*.whl"):
        process(out_dir, whl)


if __name__ == "__main__":
    main()
