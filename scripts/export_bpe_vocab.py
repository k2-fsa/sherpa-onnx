#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# You can install sentencepiece via:
#
#  pip install sentencepiece
#
# Due to an issue reported in
# https://github.com/google/sentencepiece/pull/642#issuecomment-857972030
#
# Please install a version >=0.1.96

import argparse
import codecs
from typing import Dict

try:
  import sentencepiece as spm
except ImportError:
    print('Please run')
    print('  pip install sentencepiece')
    print('before you continue')
    raise


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bpe-model",
        type=str,
        help="The path to the bpe model.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    model_file = args.bpe_model

    vocab_file = model_file.replace(".model", ".vocab")

    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
    with codecs.open(vocab_file, "w", "utf-8") as vfile:
        for v in vocabs:
            id = sp.piece_to_id(v)
            vfile.write(f"{v}\t{sp.get_score(id)}\n")
    print(f"Vocabulary file is written to {vocab_file}")


if __name__ == "__main__":
    main()
