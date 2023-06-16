from typing import Dict, List, Optional


def encode_contexts(
    modeling_unit: str,
    contexts: List[str],
    sp: Optional["SentencePieceProcessor"] = None,
    tokens_table: Optional[Dict[str, int]] = None,
) -> List[List[int]]:
    """
    Encode the given contexts (a list of string) to a list of a list of token ids.

    Args:
      modeling_unit:
        The valid values are bpe, char, bpe+char.
        Note: char here means characters in CJK languages, not English like languages.
      contexts:
        The given contexts list (a list of string).
      sp:
        An instance of SentencePieceProcessor.
      tokens_table:
        The tokens_table containing the tokens and the corresponding ids.
    Returns:
      Return the contexts_list, it is a list of a list of token ids.
    """
    contexts_list = []
    if "bpe" in modeling_unit:
        assert sp is not None
    if "char" in modeling_unit:
        assert tokens_table is not None
        assert len(tokens_table) > 0, len(tokens_table)

    if "char" == modeling_unit:
        for context in contexts:
            assert ' ' not in context
            ids = [
                tokens_table[txt] if txt in tokens_table else tokens_table["<unk>"]
                for txt in context
            ]
            contexts_list.append(ids)
    elif "bpe" == modeling_unit:
        contexts_list = sp.encode(contexts, out_type=int)
    else:
        assert modeling_unit == "bpe+char", modeling_unit

        # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
        # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        pattern = re.compile(r"([\u4e00-\u9fff])")
        for context in contexts:
            # Example:
            #   txt   = "你好 ITS'S OKAY 的"
            #   chars = ["你", "好", " ITS'S OKAY ", "的"]
            chars = pattern.split(context.upper())
            mix_chars = [w for w in chars if len(w.strip()) > 0]
            ids = []
            for ch_or_w in mix_chars:
                # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
                if pattern.fullmatch(ch_or_w) is not None:
                    ids.append(
                        tokens_table[ch_or_w]
                        if ch_or_w in tokens_table
                        else tokens_table["<unk>"]
                    )
                # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
                # encode ch_or_w using bpe_model.
                else:
                    for p in sp.encode_as_pieces(ch_or_w):
                        ids.append(
                            tokens_table[p]
                            if p in tokens_table
                            else tokens_table["<unk>"]
                        )
        contexts_list.append(ids)
    return contexts_list
