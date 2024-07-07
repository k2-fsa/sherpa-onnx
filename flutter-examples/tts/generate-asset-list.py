#!/usr/bin/env python3

"""
This file assumes that
  assets:
is the last line in ./pubspec.yaml

It read the file names of all files from the ./assets folder
and turn them as assets and write them into ./pubspec.yaml
"""

import os


def main():
    target = "./assets"
    excluded_ext = [
        ".gitkeep",
        ".onnx.json",
        ".py",
        ".sh",
        "*.md",
        "MODEL_CARD",
    ]
    sep = "    "
    ss = []
    for root, d, files in os.walk(target):
        for f in files:
            skip = False
            for p in excluded_ext:
                if f.endswith(p):
                    skip = True
                    break

            if skip:
                continue

            t = os.path.join(root, f)
            ss.append(f"{sep}- {t}")

    # read pub.spec.yaml
    with open("./pubspec.yaml", encoding="utf-8") as f:
        lines = f.readlines()

    found_assets = False
    with open("./pubspec.yaml", "w", encoding="utf-8") as f:
        for line in lines:
            if line == "  assets:\n":
                assert found_assets is False
                found_assets = True
                if len(ss) > 0:
                    f.write(line)

            if not found_assets:
                f.write(line)
                continue

            for s in ss:
                f.write(f"{s}\n")
            break

        if not found_assets and ss:
            f.write("  assets:\n")
            for s in ss:
                f.write(f"{s}\n")


if __name__ == "__main__":
    main()
