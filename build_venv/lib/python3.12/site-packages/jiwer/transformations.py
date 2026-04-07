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

"""
This file is intended to provide the default transformation which need
to be applied to input text in order to compute the WER (or similar measures).

It also implements some alternative transformations which might be
useful in specific use cases.
"""

import jiwer.transforms as tr

__all__ = [
    "wer_default",
    "wer_contiguous",
    "wer_standardize",
    "wer_standardize_contiguous",
    "cer_default",
    "cer_contiguous",
]

########################################################################################
# implement transformations for WER (and accompanying measures)

wer_default = tr.Compose(
    [
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        tr.ReduceToListOfListOfWords(),
    ]
)
"""
This is the default transformation when using `proces_words`. Each input string will 
have its leading and tailing white space removed. 
Thereafter multiple spaces between words are also removed. 
Then each string is transformed into a list with lists of strings, where each string
is a single word.
"""

wer_contiguous = tr.Compose(
    [
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        tr.ReduceToSingleSentence(),
        tr.ReduceToListOfListOfWords(),
    ]
)
"""
This is can be used instead of `wer_default` when the number of reference and hypothesis 
sentences differ. 
"""

wer_standardize = tr.Compose(
    [
        tr.ToLowerCase(),
        tr.ExpandCommonEnglishContractions(),
        tr.RemoveKaldiNonWords(),
        tr.RemoveWhiteSpace(replace_by_space=True),
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        tr.ReduceToListOfListOfWords(),
    ]
)
"""
This transform attempts to standardize the strings by setting all characters to lower
case, expanding common contractions, and removing non-words. Then the default operations
are applied.  
"""

wer_standardize_contiguous = tr.Compose(
    [
        tr.ToLowerCase(),
        tr.ExpandCommonEnglishContractions(),
        tr.RemoveKaldiNonWords(),
        tr.RemoveWhiteSpace(replace_by_space=True),
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        tr.ReduceToSingleSentence(),
        tr.ReduceToListOfListOfWords(),
    ]
)
"""
This is the same as `wer_standardize`, but this version can be usd when the number of
reference and hypothesis sentences differ. 
"""

########################################################################################
# implement transformations for CER


cer_default = tr.Compose(
    [
        tr.Strip(),
        tr.ReduceToListOfListOfChars(),
    ]
)
"""
This is the default transformation when using `process_characters`. Each input string
will  have its leading and tailing white space removed. Then each string is 
transformed into a list with lists of strings, where each string is a single character. 
"""

cer_contiguous = tr.Compose(
    [
        tr.Strip(),
        tr.ReduceToSingleSentence(),
        tr.ReduceToListOfListOfChars(),
    ]
)
"""
This can used instead of `cer_default` when the number of reference and hypothesis 
sentences differ.
"""
