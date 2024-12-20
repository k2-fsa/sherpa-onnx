#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/encoders/byte_bpe.py#L28
# and
# https://github.com/k2-fsa/icefall/blob/master/icefall/byte_utils.py
#
# Caution: The PRINTABLE_LATIN from fairseq is different from PRINTABLE_BASE_CHARS from icefall

import re

BPE_UNK = chr(8263)
PRINTABLE_BASE_CHARS = (
    list(range(256, 287 + 1))
    + list(range(32, 126 + 1))
    + list(range(288, 305 + 1))
    + list(range(308, 318 + 1))
    + list(range(321, 328 + 1))
    + list(range(330, 382 + 1))
    + list(range(384, 422 + 1))
)


BYTE_TO_BCHAR = {b: chr(PRINTABLE_BASE_CHARS[b]) for b in range(256)}
BCHAR_TO_BYTE = {bc: b for b, bc in BYTE_TO_BCHAR.items()}
BCHAR_TO_BYTE[BPE_UNK] = 32  # map unk to space


def main():
    s = ""
    s += "// sherpa-onnx/csrc/bbpe.cc\n"
    s += "//\n"
    s += "// Copyright (c)  2024 Xiaomi Corporation\n"
    s += "\n"
    s += "// Auto-generated! DO NOT EDIT\n"
    s += "\n"
    s += '#include "sherpa-onnx/csrc/bbpe.h"\n'
    s += "\n"
    s += "#include <cstdint>\n"
    s += "#include <string>\n"
    s += "#include <unordered_map>\n"
    s += "\n"
    s += "const std::unordered_map<std::string, uint8_t> &GetByteBpeTable() {\n"
    s += "  static const std::unordered_map<std::string, uint8_t> table = {\n"

    s += "      "
    for i, (k, v) in enumerate(BCHAR_TO_BYTE.items()):
        s += "{"
        if k in ["\\", '"']:
            s += f'"\{k}", {v}'
        else:
            s += f'"{k}", {v}'
        s += "}, "
        if i > 0 and i % 7 == 0:
            s += "\n"
            s += "      "
    s += "};\n"
    s += "\n"
    s += "  return table\n;"
    s += "}\n"

    with open("bbpe.cc", "w", encoding="utf-8") as f:
        f.write(s)


if __name__ == "__main__":
    main()
