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
The core algorithm(s) for processing a one or more reference and hypothesis sentences
so that measures can be computed and an alignment can be visualized.
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, List, Union

import rapidfuzz

from jiwer import transforms as tr
from jiwer.transformations import wer_default, cer_default


__all__ = [
    "AlignmentChunk",
    "WordOutput",
    "CharacterOutput",
    "process_words",
    "process_characters",
]


@dataclass
class AlignmentChunk:
    """
    Define an alignment between two subsequence of the reference and hypothesis.

    Attributes:
        type: one of `equal`, `substitute`, `insert`, or `delete`
        ref_start_idx: the start index of the reference subsequence
        ref_end_idx: the end index of the reference subsequence
        hyp_start_idx: the start index of the hypothesis subsequence
        hyp_end_idx: the end index of the hypothesis subsequence
    """

    type: str

    ref_start_idx: int
    ref_end_idx: int

    hyp_start_idx: int
    hyp_end_idx: int

    def __post_init__(self):
        if self.type not in ["replace", "insert", "delete", "equal", "substitute"]:
            raise ValueError("")

        # rapidfuzz uses replace instead of substitute... For consistency, we change it
        if self.type == "replace":
            self.type = "substitute"

        if self.ref_start_idx > self.ref_end_idx:
            raise ValueError(
                f"ref_start_idx={self.ref_start_idx} "
                f"is larger "
                f"than ref_end_idx={self.ref_end_idx}"
            )
        if self.hyp_start_idx > self.hyp_end_idx:
            raise ValueError(
                f"hyp_start_idx={self.hyp_start_idx} "
                f"is larger "
                f"than hyp_end_idx={self.hyp_end_idx}"
            )


@dataclass
class WordOutput:
    """
    The output of calculating the word-level levenshtein distance between one or more
    reference and hypothesis sentence(s).

    Attributes:
        references: The reference sentences
        hypotheses: The hypothesis sentences
        alignments: The alignment between reference and hypothesis sentences
        wer: The word error rate
        mer: The match error rate
        wil: The word information lost measure
        wip: The word information preserved measure
        hits: The number of correct words between reference and hypothesis sentences
        substitutions: The number of substitutions required to transform hypothesis
                       sentences to reference sentences
        insertions: The number of insertions required to transform hypothesis
                       sentences to reference sentences
        deletions: The number of deletions required to transform hypothesis
                       sentences to reference sentences

    """

    # processed input data
    references: List[List[str]]
    hypotheses: List[List[str]]

    # alignment
    alignments: List[List[AlignmentChunk]]

    # measures
    wer: float
    mer: float
    wil: float
    wip: float

    # stats
    hits: int
    substitutions: int
    insertions: int
    deletions: int


def process_words(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
) -> WordOutput:
    """
    Compute the word-level levenshtein distance and alignment between one or more
    reference and hypothesis sentences. Based on the result, multiple measures
    can be computed, such as the word error rate.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)

    Returns:
        (WordOutput): The processed reference and hypothesis sentences

    Raises:
        ValueError: If one or more references are empty strings
        ValueError: If after applying transforms, reference and hypothesis lengths don't match
    """
    # validate input type
    if isinstance(reference, str):
        reference = [reference]
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]

    # pre-process reference and hypothesis by applying transforms
    ref_transformed = _apply_transform(
        reference, reference_transform, is_reference=True
    )
    hyp_transformed = _apply_transform(
        hypothesis, hypothesis_transform, is_reference=False
    )

    if len(ref_transformed) != len(hyp_transformed):
        raise ValueError(
            "After applying the transforms on the reference and hypothesis sentences, "
            f"their lengths must match. "
            f"Instead got {len(ref_transformed)} reference and "
            f"{len(hyp_transformed)} hypothesis sentences."
        )

    # Map each word into a unique integer in order to compute
    # word-level levenshtein distance
    ref_as_ints, hyp_as_ints = _word2int(ref_transformed, hyp_transformed)

    # keep track of total hits, substitutions, deletions and insertions
    # across all input sentences
    num_hits, num_substitutions, num_deletions, num_insertions = 0, 0, 0, 0

    # also keep track of the total number of words in the reference and hypothesis
    num_rf_words, num_hp_words = 0, 0

    # anf finally, keep track of the alignment between each reference and hypothesis
    alignments = []

    for reference_sentence, hypothesis_sentence in zip(ref_as_ints, hyp_as_ints):
        # Get the opcodes directly
        opcodes = rapidfuzz.distance.Levenshtein.opcodes(
            reference_sentence, hypothesis_sentence
        )

        subs = dels = ins = hits = 0
        sentence_op_chunks = []

        for tag, i1, i2, j1, j2 in opcodes:
            # Create alignment chunk
            sentence_op_chunks.append(
                AlignmentChunk(
                    type=tag,
                    ref_start_idx=i1,
                    ref_end_idx=i2,
                    hyp_start_idx=j1,
                    hyp_end_idx=j2,
                )
            )

            # Update counts
            if tag == "equal":
                hits += i2 - i1
            elif tag == "replace":
                subs += i2 - i1
            elif tag == "delete":
                dels += i2 - i1
            elif tag == "insert":
                ins += j2 - j1

        # Update global counts
        num_hits += hits
        num_substitutions += subs
        num_deletions += dels
        num_insertions += ins
        num_rf_words += len(reference_sentence)
        num_hp_words += len(hypothesis_sentence)
        alignments.append(sentence_op_chunks)

    # Compute all measures
    S, D, I, H = num_substitutions, num_deletions, num_insertions, num_hits

    # special edge-case for empty references
    if num_rf_words == 0:
        wer = num_insertions

        # if the reference was silence and this is correctly predicted,
        # there is no error and all information is preserved
        if num_hp_words == 0:
            mer = 0
            wip = 1
        else:
            mer = 1
            wip = 0

    else:
        wer = float(S + D + I) / float(H + S + D)
        mer = float(S + D + I) / float(H + S + D + I)

        # there is an edge-case when hypothesis is empty
        if num_hp_words >= 1:
            wip = (float(H) / num_rf_words) * (float(H) / num_hp_words)
        else:
            wip = 0

    wil = 1 - wip

    # return all output
    return WordOutput(
        references=ref_transformed,
        hypotheses=hyp_transformed,
        alignments=alignments,
        wer=wer,
        mer=mer,
        wil=wil,
        wip=wip,
        hits=num_hits,
        substitutions=num_substitutions,
        insertions=num_insertions,
        deletions=num_deletions,
    )


########################################################################################
# Implementation of character error rate


@dataclass
class CharacterOutput:
    """
    The output of calculating the character-level levenshtein distance between one or
    more reference and hypothesis sentence(s).

    Attributes:
        references: The reference sentences
        hypotheses: The hypothesis sentences
        alignments: The alignment between reference and hypothesis sentences
        cer: The character error rate
        hits: The number of correct characters between reference and hypothesis
              sentences
        substitutions: The number of substitutions required to transform hypothesis
                       sentences to reference sentences
        insertions: The number of insertions required to transform hypothesis
                       sentences to reference sentences
        deletions: The number of deletions required to transform hypothesis
                       sentences to reference sentences
    """

    # processed input data
    references: List[List[str]]
    hypotheses: List[List[str]]

    # alignment
    alignments: List[List[AlignmentChunk]]

    # measures
    cer: float

    # stats
    hits: int
    substitutions: int
    insertions: int
    deletions: int


def process_characters(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = cer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = cer_default,
) -> CharacterOutput:
    """
    Compute the character-level levenshtein distance and alignment between one or more
    reference and hypothesis sentences. Based on the result, the character error rate
    can be computed.

    Note that the by default this method includes space (` `) as a
    character over which the error rate is computed. If this is not desired, the
    reference and hypothesis transform need to be modified.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)

    Returns:
        (CharacterOutput): The processed reference and hypothesis sentences.

    """
    # it's the same as word processing, just every word is of length 1
    result = process_words(
        reference, hypothesis, reference_transform, hypothesis_transform
    )

    return CharacterOutput(
        references=result.references,
        hypotheses=result.hypotheses,
        alignments=result.alignments,
        cer=result.wer,
        hits=result.hits,
        substitutions=result.substitutions,
        insertions=result.insertions,
        deletions=result.deletions,
    )


################################################################################
# Implementation of helper methods


def _apply_transform(
    sentence: Union[str, List[str]],
    transform: Union[tr.Compose, tr.AbstractTransform],
    is_reference: bool,
):
    # Apply transforms. The transforms should collapse input to a
    # list with lists of words
    transformed_sentence = transform(sentence)

    # Validate the output is a list containing lists of strings
    if not _is_list_of_list_of_strings(transformed_sentence):
        raise ValueError(
            "After applying the transformation, each "
            f"{'reference' if is_reference else 'hypothesis'} should be a "
            "list of strings, with each string being a single word or character."
            "Please ensure the given transformation reduces the input "
            "to a list of list strings."
        )

    return transformed_sentence


def _is_list_of_list_of_strings(x: Any):
    if not isinstance(x, list):
        return False

    for e in x:
        if not isinstance(e, list):
            return False

        if not all([isinstance(s, str) for s in e]):
            return False

    return True


def _word2int(reference: List[List[str]], hypothesis: List[List[str]]):
    """
    Maps each unique word in the reference and hypothesis sentences to a unique integer
    for Levenshtein distance calculation.

    Args:
        reference: List of reference sentences, where each sentence is a list of words
        hypothesis: List of hypothesis sentences, where each sentence is a list of words

    Returns:
        Tuple[List[List[int]], List[List[int]]]: The reference and hypothesis sentences
        with words mapped to unique integers
    """
    word2int = defaultdict()
    word2int.default_factory = word2int.__len__  # Auto-incrementing IDs

    # Single pass through all words using generator expressions
    ref_ints = [[word2int[word] for word in sentence] for sentence in reference]
    hyp_ints = [[word2int[word] for word in sentence] for sentence in hypothesis]

    return ref_ints, hyp_ints
