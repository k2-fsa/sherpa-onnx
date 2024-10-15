#!/usr/bin/env python3

"""
This file assumes that
  assets:
is the last line in ./pubspec.yaml

It reads the file names of all files from the ./assets folder
and turns them as assets and writes them into ./pubspec.yaml
"""

import os

def main():
    target = "./assets/"
    space = "    "
    subfolders = []
    patterns_to_skip = ["1.5x", "2.x", "3.x", "4.x"]
    for root, dirs, files in os.walk(target):
        for d in dirs:
            path = os.path.join(root, d).replace("\\", "/")
            if os.listdir(path):
                path = path.lstrip('./')
                if any(path.endswith(pattern) for pattern in patterns_to_skip):
                    continue
                subfolders.append("{space}- {path}/".format(space=space, path=path))

    assert subfolders, "The subfolders list is empty."

    subfolders = sorted(subfolders)

    loc_of_flutter = -1
    loc_of_flutter_asset = -1
    loc_of_end_flutter_asset = -1
    loc_of_end_flutter = -1

    with open("./pubspec.yaml", encoding="utf-8") as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if line == "flutter:\n":
                loc_of_flutter = index + 1
                if index == len(lines) - 1:
                    loc_of_end_flutter = index + 2
                continue
            if loc_of_flutter >= 0 and loc_of_flutter_asset < 0 and line == "  assets:\n":
                loc_of_flutter_asset = index + 1
                continue

    with open("./pubspec.yaml", encoding="utf-8") as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if index < loc_of_flutter:
                continue
            if loc_of_flutter_asset >= 0:
                if line.startswith("    - assets/"):
                    loc_of_end_flutter_asset = index + 1
                    continue
                else:
                    loc_of_end_flutter = index + 1
                    continue
            else:
                if line.startswith("  ") is False:
                    loc_of_end_flutter = index + 1
                    continue
                else:
                    loc_of_end_flutter = index + 2
                    break

    assert loc_of_flutter >= 0, "The 'flutter:' section is missing in the pubspec.yaml file."

    with open("./pubspec.yaml", "w", encoding="utf-8") as f:
        for index, line in enumerate(lines):
            if loc_of_end_flutter_asset >= 0:
                if index + 1 < loc_of_flutter_asset or index + 1 > loc_of_end_flutter_asset:
                    f.write(line)
                if index + 1 == loc_of_flutter_asset:
                    f.write("  assets:\n")
                    for folder in subfolders:
                        f.write("{folder}\n".format(folder=folder))
            else:
                if index + 1 < loc_of_end_flutter or index + 1 > loc_of_end_flutter:
                    f.write(line)
                if index + 1 == loc_of_end_flutter:
                    f.write("  assets:\n")
                    for indexOfFolder, folder in enumerate(subfolders):
                        f.write("{folder}\n".format(folder=folder))
                        if indexOfFolder == len(subfolders) - 1:
                            f.write("\n")
                            break

        if loc_of_end_flutter == len(lines) + 1:
            f.write("\n")
            f.write("  assets:\n")
            for folder in subfolders:
                f.write("{folder}\n".format(folder=folder))

if __name__ == "__main__":
    main()
