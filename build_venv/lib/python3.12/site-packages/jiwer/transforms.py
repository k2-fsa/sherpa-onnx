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
This file implements the building blocks for transforming a collection
of input strings to the desired format in order to calculate the WER of CER.

In principle, for word error rate calculations, every string of a sentence needs to be
collapsed into a list of strings, where each string is a *single* word.
This is done with [transforms.ReduceToListOfListOfWords][].
A composition of multiple transformations must therefore *always* end with
[transforms.ReduceToListOfListOfWords][].

For the character error rate, every string of a sentence also needs to be collapsed into
a list of strings, but here each string is a *single* character.
This is done with [transforms.ReduceToListOfListOfChars][]. Similarly, a
composition of multiple transformations must therefore also always end with
[transforms.ReduceToListOfListOfChars][].
"""

import sys
import functools
import re
import string
import unicodedata

from typing import Iterable, Union, List, Mapping


__all__ = [
    "AbstractTransform",
    "Compose",
    "ExpandCommonEnglishContractions",
    "RemoveEmptyStrings",
    "ReduceToListOfListOfWords",
    "ReduceToListOfListOfChars",
    "ReduceToSingleSentence",
    "RemoveKaldiNonWords",
    "RemoveMultipleSpaces",
    "RemovePunctuation",
    "RemoveSpecificWords",
    "RemoveWhiteSpace",
    "Strip",
    "SubstituteRegexes",
    "SubstituteWords",
    "ToLowerCase",
    "ToUpperCase",
]


class AbstractTransform(object):
    """
    The base class of a Transform.
    """

    def __call__(self, sentences: Union[str, List[str]]):
        """
        Transforms one or more strings.

        Args:
            sentences: The strings to transform.

        Returns:
            (Union[str, List[str]]): The transformed strings.

        """
        if isinstance(sentences, str):
            return self.process_string(sentences)
        elif isinstance(sentences, list):
            return self.process_list(sentences)
        else:
            raise ValueError(
                "input {} was expected to be a string or list of strings".format(
                    sentences
                )
            )

    def process_string(self, s: str):
        raise NotImplementedError()

    def process_list(self, inp: List[str]):
        return [self.process_string(s) for s in inp]


class Compose(object):
    """
    Chain multiple transformations back-to-back to create a pipeline combining multiple
    transformations.

    Note that each transformation needs to end with either `ReduceToListOfListOfWords`
    or `ReduceToListOfListOfChars`, depending on whether word error rate,
    or character error rate is desired.

    Example:
        ```python3
        import jiwer

        jiwer.Compose([
            jiwer.RemoveMultipleSpaces(),
            jiwer.ReduceToListOfListOfWords()
        ])
        ```
    """

    def __init__(self, transforms: List[AbstractTransform]):
        """

        Args:
            transforms: The list of transformations to chain.
        """
        self.transforms = transforms

    def __call__(self, text):
        for tr in self.transforms:
            text = tr(text)

        return text


class BaseRemoveTransform(AbstractTransform):
    def __init__(self, tokens_to_remove: Iterable[str], replace_token=""):
        self.tokens_to_remove = tokens_to_remove
        self.replace_token = replace_token

    def process_string(self, s: str):
        for w in self.tokens_to_remove:
            s = s.replace(w, self.replace_token)

        return s

    def process_list(self, inp: List[str]):
        return [self.process_string(s) for s in inp]


class ReduceToListOfListOfWords(AbstractTransform):
    """
    Transforms a single input sentence, or a list of input sentences, into
    a list with lists of words, which is the expected format for calculating the
    edit operations between two input sentences on a word-level.

    A sentence is assumed to be a string, where words are delimited by a token
    (such as ` `, space). Each string is expected to contain only a single sentence.
    Empty strings (no output) are removed for the list.

    Example:
        ```python
        import jiwer

        sentences = ["hi", "this is an example"]

        print(jiwer.ReduceToListOfListOfWords()(sentences))
        # prints: [['hi'], ['this', 'is', 'an, 'example']]
        ```
    """

    def __init__(self, word_delimiter: str = " "):
        """
        Args:
            word_delimiter: the character which delimits words. Default is ` ` (space).
        """
        self.word_delimiter = word_delimiter

    def process_string(self, s: str):
        return [[w for w in s.split(self.word_delimiter) if len(w) >= 1]]

    def process_list(self, inp: List[str]):
        sentence_collection = []

        for sentence in inp:
            list_of_words = self.process_string(sentence)[0]
            sentence_collection.append(list_of_words)

        if len(sentence_collection) == 0:
            return [[]]

        return sentence_collection


class ReduceToListOfListOfChars(AbstractTransform):
    """
    Transforms a single input sentence, or a list of input sentences, into
    a list with lists of characters, which is the expected format for calculating the
    edit operations between two input sentences on a character-level.

    A sentence is assumed to be a string. Each string is expected to contain only a
    single sentence.

    Example:
        ```python
        import jiwer

        sentences = ["hi", "this is an example"]

        print(jiwer.ReduceToListOfListOfChars()(sentences))
        # prints: [['h', 'i'], ['t', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', 'n', ' ', 'e', 'x', 'a', 'm', 'p', 'l', 'e']]
        ```
    """

    def process_string(self, s: str):
        return [[w for w in s]]

    def process_list(self, inp: List[str]):
        sentence_collection = []

        for sentence in inp:
            list_of_words = self.process_string(sentence)[0]

            sentence_collection.append(list_of_words)

        if len(sentence_collection) == 0:
            return [[]]

        return sentence_collection


class ReduceToSingleSentence(AbstractTransform):
    """
    Transforms multiple sentences into a single sentence.
    This operation can be useful when the number of reference and hypothesis sentences
    differ, and you want to do a minimal alignment over these lists.
    Note that this creates an invariance: `wer([a, b], [a, b])` might not be equal to
    `wer([b, a], [b, a])`.

    Example:
        ```python3
        import jiwer

        sentences = ["hi", "this is an example"]

        print(jiwer.ReduceToSingleSentence()(sentences))
        # prints: ['hi this is an example']
        ```
    """

    def __init__(self, word_delimiter: str = " "):
        """
        :param word_delimiter: the character which delimits words. Default is ` ` (space).
        """
        self.word_delimiter = word_delimiter

    def process_string(self, s: str):
        return s

    def process_list(self, inp: List[str]):
        filtered_inp = [i for i in inp if len(i) >= 1]

        if len(filtered_inp) == 0:
            return []
        else:
            return ["{}".format(self.word_delimiter).join(filtered_inp)]


class SubstituteRegexes(AbstractTransform):
    r"""
    Transform strings by substituting substrings matching regex expressions into
    another substring.

    Example:
        ```python
        import jiwer

        sentences = ["is the world doomed or loved?", "edibles are allegedly cultivated"]

        # note: the regex string "\b(\w+)ed\b", matches every word ending in 'ed',
        # and "\1" stands for the first group ('\w+). It therefore removes 'ed' in every match.
        print(jiwer.SubstituteRegexes({r"doom": r"sacr", r"\b(\w+)ed\b": r"\1"})(sentences))

        # prints: ["is the world sacr or lov?", "edibles are allegedly cultivat"]
        ```
    """

    def __init__(self, substitutions: Mapping[str, str]):
        """

        Args:
            substitutions: a mapping of regex expressions to replacement strings.
        """
        self.substitutions = substitutions

    def process_string(self, s: str):
        for key, value in self.substitutions.items():
            s = re.sub(key, value, s)

        return s


class SubstituteWords(AbstractTransform):
    """
    This transform can be used to replace a word into another word.
    Note that the whole word is matched. If the word you're attempting to substitute
    is a substring of another word it will not be affected.
    For example, if you're substituting `foo` into `bar`, the word `foobar` will NOT
    be substituted into `barbar`.

    Example:
        ```python
        import jiwer

        sentences = ["you're pretty", "your book", "foobar"]

        print(jiwer.SubstituteWords({"pretty": "awesome", "you": "i", "'re": " am", 'foo': 'bar'})(sentences))

        # prints: ["i am awesome", "your book", "foobar"]
        ```

    """

    def __init__(self, substitutions: Mapping[str, str]):
        """
        Args:
            substitutions: A mapping of words to replacement words.
        """
        self.substitutions = substitutions

    def process_string(self, s: str):
        for key, value in self.substitutions.items():
            s = re.sub(r"\b{}\b".format(re.escape(key)), value, s)

        return s


class RemoveSpecificWords(SubstituteWords):
    """
    Can be used to filter out certain words.
    As words are replaced with a ` ` character, make sure to that
    `RemoveMultipleSpaces`, `Strip()` and `RemoveEmptyStrings` are present
    in the composition _after_ `RemoveSpecificWords`.

    Example:
        ```python
        import jiwer

        sentences = ["yhe awesome", "the apple is not a pear", "yhe"]

        print(jiwer.RemoveSpecificWords(["yhe", "the", "a"])(sentences))
        # prints: ['  awesome', '  apple is not   pear', ' ']
        # note the extra spaces
        ```
    """

    def __init__(self, words_to_remove: List[str]):
        """
        Args:
            words_to_remove: List of words to remove.
        """
        mapping = {word: " " for word in words_to_remove}

        super().__init__(mapping)


class RemoveWhiteSpace(BaseRemoveTransform):
    """
    This transform filters out white space characters.
    Note that by default space (` `) is also removed, which will make it impossible to
    split a sentence into a list of words by using `ReduceToListOfListOfWords` or
    `ReduceToSingleSentence`.
    This can be prevented by replacing all whitespace with the space character.
    If so, make sure that `jiwer.RemoveMultipleSpaces`,
    `Strip()` and `RemoveEmptyStrings` are present in the composition _after_
    `RemoveWhiteSpace`.

    Example:
        ```python
        import jiwer

        sentences = ["this is an example", "hello world\t"]

        print(jiwer.RemoveWhiteSpace()(sentences))
        # prints: ["thisisanexample", "helloworld"]

        print(jiwer.RemoveWhiteSpace(replace_by_space=True)(sentences))
        # prints: ["this is an example", "hello world  "]
        # note the trailing spaces
        ```
    """

    def __init__(self, replace_by_space: bool = False):
        """

        Args:
            replace_by_space: every white space character is replaced with a space (` `)
        """
        characters = [c for c in string.whitespace]

        if replace_by_space:
            replace_token = " "
        else:
            replace_token = ""

        super().__init__(characters, replace_token=replace_token)


@functools.lru_cache(1)
def _get_punctuation_characters():
    """Compute the punctuation characters only once and memoize."""
    codepoints = range(sys.maxunicode + 1)
    punctuation = set(
        chr(i) for i in codepoints if unicodedata.category(chr(i)).startswith("P")
    )
    return punctuation


class RemovePunctuation(BaseRemoveTransform):
    """
    This transform filters out punctuation. The punctuation characters are defined as
    all unicode characters whose category name starts with `P`.
    See [here](https://www.unicode.org/reports/tr44/#General_Category_Values) for more
    information.
    Example:
        ```python
        import jiwer

        sentences = ["this is an example!", "hello. goodbye"]

        print(jiwer.RemovePunctuation()(sentences))
        # prints: ['this is an example', "hello goodbye"]
        ```
    """

    def __init__(self):
        punctuation_characters = _get_punctuation_characters()
        super().__init__(punctuation_characters)


class RemoveMultipleSpaces(AbstractTransform):
    """
    Filter out multiple spaces between words.

    Example:
        ```python
        import jiwer

        sentences = ["this is   an   example ", "  hello goodbye  ", "  "]

        print(jiwer.RemoveMultipleSpaces()(sentences))
        # prints: ['this is an example ', " hello goodbye ", " "]
        # note that there are still trailing spaces
        ```

    """

    def process_string(self, s: str):
        return re.sub(r"\s\s+", " ", s)

    def process_list(self, inp: List[str]):
        return [self.process_string(s) for s in inp]


class Strip(AbstractTransform):
    """
    Removes all leading and trailing spaces.

    Example:
        ```python
        import jiwer

        sentences = [" this is an example ", "  hello goodbye  ", "  "]

        print(jiwer.Strip()(sentences))
        # prints: ['this is an example', "hello goodbye", ""]
        # note that there is an empty string left behind which might need to be cleaned up
        ```
    """

    def process_string(self, s: str):
        return s.strip()


class RemoveEmptyStrings(AbstractTransform):
    """
    Remove empty strings from a list of strings.

    Example:
        ```python
        import jiwer

        sentences = ["", "this is an example", " ",  "                "]

        print(jiwer.RemoveEmptyStrings()(sentences))
        # prints: ['this is an example']
        ```
    """

    def process_string(self, s: str):
        return s.strip()

    def process_list(self, inp: List[str]):
        return [s for s in inp if self.process_string(s) != ""]


class ExpandCommonEnglishContractions(AbstractTransform):
    """
    Replace common contractions such as `let's` to `let us`.

    Currently, this method will perform the following replacements. Note that `␣` is
     used to indicate a space (` `) to get around markdown rendering constrains.

    | Contraction   | transformed into |
    | ------------- |:----------------:|
    | `won't`       | `␣will not`      |
    | `can't`       | `␣can not`       |
    | `let's`       | `␣let us`        |
    | `n't`         | `␣not`           |
    | `'re`         | `␣are`           |
    | `'s`          | `␣is`            |
    | `'d`          | `␣would`         |
    | `'ll`         | `␣will`          |
    | `'t`          | `␣not`           |
    | `'ve`         | `␣have`          |
    | `'m`          | `␣am`            |

    Example:
        ```python
        import jiwer

        sentences = ["she'll make sure you can't make it", "let's party!"]

        print(jiwer.ExpandCommonEnglishContractions()(sentences))
        # prints: ["she will make sure you can not make it", "let us party!"]
        ```

    """

    def process_string(self, s: str):
        # definitely a non exhaustive list

        # specific words
        s = re.sub(r"won't", "will not", s)
        s = re.sub(r"can\'t", "can not", s)
        s = re.sub(r"let\'s", "let us", s)

        # general attachments
        s = re.sub(r"n\'t", " not", s)
        s = re.sub(r"\'re", " are", s)
        s = re.sub(r"\'s", " is", s)
        s = re.sub(r"\'d", " would", s)
        s = re.sub(r"\'ll", " will", s)
        s = re.sub(r"\'t", " not", s)
        s = re.sub(r"\'ve", " have", s)
        s = re.sub(r"\'m", " am", s)

        return s


class ToLowerCase(AbstractTransform):
    """
    Convert every character into lowercase.
    Example:
        ```python
        import jiwer

        sentences = ["You're PRETTY"]

        print(jiwer.ToLowerCase()(sentences))

        # prints: ["you're pretty"]
        ```
    """

    def process_string(self, s: str):
        return s.lower()


class ToUpperCase(AbstractTransform):
    """
    Convert every character to uppercase.

    Example:
        ```python
        import jiwer

        sentences = ["You're amazing"]

        print(jiwer.ToUpperCase()(sentences))

        # prints: ["YOU'RE AMAZING"]
        ```
    """

    def process_string(self, s: str):
        return s.upper()


class RemoveKaldiNonWords(AbstractTransform):
    """
    Remove any word between `[]` and `<>`. This can be useful when working
    with hypotheses from the Kaldi project, which can output non-words such as
    `[laugh]` and `<unk>`.

    Example:
        ```python
        import jiwer

        sentences = ["you <unk> like [laugh]"]

        print(jiwer.RemoveKaldiNonWords()(sentences))

        # prints: ["you  like "]
        # note the extra spaces
        ```
    """

    def process_string(self, s: str):
        return re.sub(r"[<\[][^>\]]*[>\]]", "", s)
