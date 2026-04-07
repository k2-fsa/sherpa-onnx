#
# JiWER - Jitsi Word Error Rate
#
# Copyright @ 2018 - present 8x8, Inc.
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
#

"""
Provide a simple CLI wrapper for JiWER. The CLI does not support custom transforms.
"""

import click
import pathlib

import jiwer


@click.command()
@click.option(
    "-r",
    "--reference",
    "reference_file",
    type=pathlib.Path,
    required=True,
    help="Path to new-line delimited text file of reference sentences.",
)
@click.option(
    "-h",
    "--hypothesis",
    "hypothesis_file",
    type=pathlib.Path,
    required=True,
    help="Path to new-line delimited text file of hypothesis sentences.",
)
@click.option(
    "--cer",
    "-c",
    "compute_cer",
    is_flag=True,
    default=False,
    help="Compute CER instead of WER.",
)
@click.option(
    "--align",
    "-a",
    "show_alignment",
    is_flag=True,
    default=False,
    help="Print alignment of each sentence.",
)
@click.option(
    "--global",
    "-g",
    "global_alignment",
    is_flag=True,
    default=False,
    help="Apply a global minimal alignment between reference and hypothesis sentences "
    "before computing the WER.",
)
def cli(
    reference_file: pathlib.Path,
    hypothesis_file: pathlib.Path,
    compute_cer: bool,
    show_alignment: bool,
    global_alignment: bool,
):
    """
    JiWER is a python tool for computing the word-error-rate of ASR systems. To use
    this CLI, store the reference and hypothesis sentences in a text file, where
    each sentence is delimited by a new-line character.
    The text files are expected to have an equal number of lines, unless the `-g` flag
    is used. The `-g` flag joins computation of the WER by doing a global minimal
    alignment.

    """
    with reference_file.open("r") as f:
        reference_sentences = [
            ln.strip() for ln in f.readlines() if len(ln.strip()) > 1
        ]

    with hypothesis_file.open("r") as f:
        hypothesis_sentences = [
            ln.strip() for ln in f.readlines() if len(ln.strip()) > 1
        ]

    if not global_alignment and len(reference_sentences) != len(hypothesis_sentences):
        raise ValueError(
            f"Number of reference sentences "
            f"({len(reference_sentences)} in '{reference_file}') "
            f"and hypothesis sentences "
            f"({len(hypothesis_sentences)} in '{hypothesis_file}') "
            f"do not match! "
            f"Use the `--global` flag to compute the measures over a global alignment "
            f"of the reference and hypothesis sentences."
        )

    if compute_cer:
        if global_alignment:
            out = jiwer.process_characters(
                reference_sentences,
                hypothesis_sentences,
                reference_transform=jiwer.cer_contiguous,
                hypothesis_transform=jiwer.cer_contiguous,
            )
        else:
            out = jiwer.process_characters(
                reference_sentences,
                hypothesis_sentences,
            )
    else:
        if global_alignment:
            out = jiwer.process_words(
                reference_sentences,
                hypothesis_sentences,
                reference_transform=jiwer.wer_contiguous,
                hypothesis_transform=jiwer.wer_contiguous,
            )
        else:
            out = jiwer.process_words(reference_sentences, hypothesis_sentences)

    if show_alignment:
        print(jiwer.visualize_alignment(out, show_measures=True), end="")
    else:
        if compute_cer:
            print(out.cer)
        else:
            print(out.wer)


if __name__ == "__main__":
    cli()
