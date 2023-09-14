# Copyright (c)  2023  Xiaomi Corporation
import re

from pathlib import Path
from typing import List, Optional, Union

import sentencepiece as spm


def text2token(
    texts: List[str],
    tokens: str,
    tokens_type: str = "cjkchar",
    bpe_model: Optional[str] = None,
    output_ids: bool = False,
) -> List[List[Union[str, int]]]:
    """
    Encode the given texts (a list of string) to a list of a list of tokens.

    Args:
      texts:
        The given contexts list (a list of string).
      tokens:
        The path of the tokens.txt.
      tokens_type:
        The valid values are cjkchar, bpe, cjkchar+bpe.
      bpe_model:
        The path of the bpe model. Only required when tokens_type is bpe or
        cjkchar+bpe.
      output_ids:
        True to output token ids otherwise tokens.
    Returns:
      Return the encoded texts, it is a list of a list of token ids if output_ids
      is True, or it is a list of list of tokens.
    """
    assert Path(tokens).is_file(), f"File not exists, {tokens}"
    tokens_table = {}
    with open(tokens, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            assert len(toks) == 2, len(toks)
            assert toks[0] not in tokens_table, f"Duplicate token: {toks} "
            tokens_table[toks[0]] = int(toks[1])

    if "bpe" in tokens_type:
        assert Path(bpe_model).is_file(), f"File not exists, {bpe_model}"
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    texts_list: List[List[str]] = []

    if tokens_type == "cjkchar":
        texts_list = [list("".join(text.split())) for text in texts]
    elif tokens_type == "bpe":
        texts_list = sp.encode(texts, out_type=str)
    else:
        assert (
            tokens_type == "cjkchar+bpe"
        ), f"Supported tokens_type are cjkchar, bpe, cjkchar+bpe, given {tokens_type}"
        # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
        # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        pattern = re.compile(r"([\u4e00-\u9fff])")
        for text in texts:
            # Example:
            #   txt   = "你好 ITS'S OKAY 的"
            #   chars = ["你", "好", " ITS'S OKAY ", "的"]
            chars = pattern.split(text)
            mix_chars = [w for w in chars if len(w.strip()) > 0]
            text_list = []
            for ch_or_w in mix_chars:
                # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
                if pattern.fullmatch(ch_or_w) is not None:
                    text_list.append(ch_or_w)
                # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
                # encode ch_or_w using bpe_model.
                else:
                    text_list += sp.encode_as_pieces(ch_or_w)
            texts_list.append(text_list)

    result: List[List[Union[int, str]]] = []
    for text in texts_list:
        text_list = []
        contain_oov = False
        for txt in text:
            if txt in tokens_table:
                text_list.append(tokens_table[txt] if output_ids else txt)
            else:
                print(f"OOV token : {txt}, skipping text : {text}.")
                contain_oov = True
                break
        if contain_oov:
            continue
        else:
            result.append(text_list)
    return result
