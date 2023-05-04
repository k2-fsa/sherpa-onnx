# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                    Mingshuang Luo,
#                                                    Zengwei Yao)
#
# See ../../LICENSE for clarification regarding multiple authors
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

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple, Union

import kaldialign

Pathlike = Union[str, Path]


def store_transcripts(
    filename: Pathlike, texts: Iterable[Tuple[str, str, str]]
) -> None:
    """Save predicted results and reference transcripts to a file.

    Args:
      filename:
        File to save the results to.
      texts:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
    Returns:
      Return None.
    """
    with open(filename, "w") as f:
        for cut_id, ref, hyp in texts:
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)


def write_error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str]],
    enable_log: bool = True,
) -> float:
    """Write statistics based on predicted results and reference transcripts.

    It will write the following to the given file:

        - WER
        - number of insertions, deletions, substitutions, corrects and total
          reference words. For example::

              Errors: 23 insertions, 57 deletions, 212 substitutions, over 2606
              reference words (2337 correct)

        - The difference between the reference transcript and predicted result.
          An instance is given below::

            THE ASSOCIATION OF (EDISON->ADDISON) ILLUMINATING COMPANIES

          The above example shows that the reference word is `EDISON`,
          but it is predicted to `ADDISON` (a substitution error).

          Another example is::

            FOR THE FIRST DAY (SIR->*) I THINK

          The reference word `SIR` is missing in the predicted
          results (a deletion error).
      results:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
      enable_log:
        If True, also print detailed WER to the console.
        Otherwise, it is written only to the given file.
    Returns:
      Return None.
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
    ref_len = sum([len(r) for _, r, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)

    if enable_log:
        logging.info(
            f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} "
            f"[{tot_errs} / {ref_len}, {ins_errs} ins, "
            f"{del_errs} del, {sub_errs} sub ]"
        )

    print(f"%WER = {tot_err_rate}", file=f)
    print(
        f"Errors: {ins_errs} insertions, {del_errs} deletions, "
        f"{sub_errs} substitutions, over {ref_len} reference "
        f"words ({num_corr} correct)",
        file=f,
    )
    print(
        "Search below for sections starting with PER-UTT DETAILS:, "
        "SUBSTITUTIONS:, DELETIONS:, INSERTIONS:, PER-WORD STATS:",
        file=f,
    )

    print("", file=f)
    print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        combine_successive_errors = True
        if combine_successive_errors:
            ali = [[[x], [y]] for x, y in ali]
            for i in range(len(ali) - 1):
                if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
                    ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
                    ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
                    ali[i] = [[], []]
            ali = [
                [
                    list(filter(lambda a: a != ERR, x)),
                    list(filter(lambda a: a != ERR, y)),
                ]
                for x, y in ali
            ]
            ali = list(filter(lambda x: x != [[], []], ali))
            ali = [
                [
                    ERR if x == [] else " ".join(x),
                    ERR if y == [] else " ".join(y),
                ]
                for x, y in ali
            ]

        print(
            f"{cut_id}:\t"
            + " ".join(
                (
                    ref_word if ref_word == hyp_word else f"({ref_word}->{hyp_word})"
                    for ref_word, hyp_word in ali
                )
            ),
            file=f,
        )

    print("", file=f)
    print("SUBSTITUTIONS: count ref -> hyp", file=f)

    for count, (ref, hyp) in sorted([(v, k) for k, v in subs.items()], reverse=True):
        print(f"{count}   {ref} -> {hyp}", file=f)

    print("", file=f)
    print("DELETIONS: count ref", file=f)
    for count, ref in sorted([(v, k) for k, v in dels.items()], reverse=True):
        print(f"{count}   {ref}", file=f)

    print("", file=f)
    print("INSERTIONS: count hyp", file=f)
    for count, hyp in sorted([(v, k) for k, v in ins.items()], reverse=True):
        print(f"{count}   {hyp}", file=f)

    print("", file=f)
    print("PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp", file=f)
    for _, word, counts in sorted(
        [(sum(v[1:]), k, v) for k, v in words.items()], reverse=True
    ):
        (corr, ref_sub, hyp_sub, ins, dels) = counts
        tot_errs = ref_sub + hyp_sub + ins + dels
        ref_count = corr + ref_sub + dels
        hyp_count = corr + hyp_sub + ins

        print(f"{word}   {corr} {tot_errs} {ref_count} {hyp_count}", file=f)
    return float(tot_err_rate)


def write_triton_stats(stats, summary_file):
    with open(summary_file, "w") as summary_f:
        model_stats = stats["model_stats"]
        for model_state in model_stats:
            if "last_inference" not in model_state:
                continue
            summary_f.write(f"model name is {model_state['name']} \n")
            model_inference_stats = model_state["inference_stats"]
            total_queue_time_s = int(model_inference_stats["queue"]["ns"]) / 1e9
            total_infer_time_s = int(model_inference_stats["compute_infer"]["ns"]) / 1e9
            total_input_time_s = int(model_inference_stats["compute_input"]["ns"]) / 1e9
            total_output_time_s = (
                int(model_inference_stats["compute_output"]["ns"]) / 1e9
            )
            summary_f.write(
                f"queue {total_queue_time_s:<5.2f} s, infer {total_infer_time_s:<5.2f} s, input {total_input_time_s:<5.2f} s, output {total_output_time_s:<5.2f} s \n"  # noqa
            )
            model_batch_stats = model_state["batch_stats"]
            for batch in model_batch_stats:
                batch_size = int(batch["batch_size"])
                compute_input = batch["compute_input"]
                compute_output = batch["compute_output"]
                compute_infer = batch["compute_infer"]
                batch_count = int(compute_infer["count"])
                assert (
                    compute_infer["count"]
                    == compute_output["count"]
                    == compute_input["count"]
                )
                compute_infer_time_ms = int(compute_infer["ns"]) / 1e6
                compute_input_time_ms = int(compute_input["ns"]) / 1e6
                compute_output_time_ms = int(compute_output["ns"]) / 1e6
                summary_f.write(
                    f"Batch_size {batch_size:<2}, {batch_count:<5} times, infer {compute_infer_time_ms:<9.2f} ms, avg {compute_infer_time_ms/batch_count:.2f} ms, {compute_infer_time_ms/batch_count/batch_size:.2f} ms "  # noqa
                )
                summary_f.write(
                    f"input {compute_input_time_ms:<9.2f} ms, avg {compute_input_time_ms/batch_count:.2f} ms, "  # noqa
                )
                summary_f.write(
                    f"output {compute_output_time_ms:<9.2f} ms, avg {compute_output_time_ms/batch_count:.2f} ms \n"  # noqa
                )


def download_and_extract(
    target_path: str,
    url: str = "https://huggingface.co/csukuangfj/aishell-test-dev-manifests/reslove/main/data_aishell.tar.gz",
):
    filename = url.split("/")[-1]

    # Download file using wget if it doesn't exist
    command = f"wget -nc {url}"
    os.system(command)

    # Extract file using tar
    command = f"tar -xf {filename} -C {target_path}"
    os.system(command)

    # Delete downloaded file
    os.remove(filename)
