import unicodedata
from indic_numtowords.san.data.nums import DIRECT_DICT
from indic_numtowords.san.data.nums import NUMBER_SCALE_DICT
from indic_numtowords.san.data.nums import PREFIX_REPLACEMENTS
from indic_numtowords.san.data.nums import VARIATIONS_DICT
from indic_numtowords.san.utils import combine

def combine_morphemes(primary_text: str, secondary_text: str) -> str:
  """
    Combine two morphemes according to Sanskrit samasa rules.

    Args:
        primary_text (str): The primary text to be combined.
        secondary_text (str): The secondary text to be attached to the primary.

    Returns:
        str: The resulting combined text with a normalized Unicode representation.
    """
  suffix = next((key for key in PREFIX_REPLACEMENTS if primary_text.endswith(key)), None)
  if suffix:
    for suffixes, replacement in PREFIX_REPLACEMENTS[suffix].items():
      matched_prefix = next((val for val in suffixes if secondary_text.startswith(val)), None)
      if matched_prefix:
        primary_text = primary_text[:-len(suffix)] + PREFIX_REPLACEMENTS[suffix][suffixes]
        if (matched_prefix in ['द', 'वि', 'त्रि', 'च', 'प', 'ष', 'स', 'न', 'ल', 'श', 'को', 'ख', 'म', 'ज'] and suffix =='एकं') or (matched_prefix in ['अ', 'उ']) or (matched_prefix == 'द' and suffix == 'षट्'):
          secondary_text = secondary_text[len(matched_prefix):]
        break

  combined = primary_text + secondary_text
  normalized = unicodedata.normalize('NFC', combined)
  return normalized

def convert_to_text(number_str: str, index: int, number_len: int) -> list[str] | tuple[list[str], list[str]]:
    """
    Convert a number to its text representation.

    Args:
        number_str (str): The number to convert.

    Returns:
        list[str]: The text representation of the number.
    """
    texts = []
    prefixes = []

    suffix = 'उत्तर' if number_len > 3 and len(number_str) == 3 else 'अधिक' if number_len > 3 and index == 0 else None

    # Handle three-digit numbers
    if len(number_str) == 3:
      if number_str[0]!='1':
        prefixes.extend(DIRECT_DICT[number_str[0]])
        prefixes.extend(variation for variation in VARIATIONS_DICT.get(number_str[0], []))

      number_str = number_str[1:].lstrip('0')
      suffix = 'उत्तर' if number_len > 3 else 'अधिक'
      texts.extend([combine_morphemes(DIRECT_DICT[number_str][0], suffix)] if number_str else [])
      texts.extend([combine_morphemes(variation, suffix) for variation in VARIATIONS_DICT.get(number_str, [])])
      return prefixes, texts

    # Handle one or two-digit numbers
    else:
      if suffix and index==0:
        texts.extend([combine_morphemes(DIRECT_DICT[number_str][0], suffix)])
        texts.extend([combine_morphemes(variation, suffix) for variation in VARIATIONS_DICT.get(number_str, [])])
      else:
        texts.extend(DIRECT_DICT[number_str])
        texts.extend(variation for variation in VARIATIONS_DICT.get(number_str, []))
    return texts


def process_text(number_str: str, texts: list[str], index: int, number_len) -> list[str]:
    """
    Convert a number string to its text representation and append it to a list.

    Args:
        number_str (str): The number string to process.
        texts (list[str]): A list of texts to update with the converted number.
        index (int): The position of the number in the sequence, affecting scaling.
        number_len (int): The total length of the original number.

    Returns:
        list[str]: The updated list with the number's text representation appended.
    """
    if not isinstance(number_str, str) or not isinstance(index, int):
        raise ValueError("Invalid input type")

    if number_str in {'00', '000'} or (index > 0 and number_str == '0'):
        return texts

    number_str = number_str.lstrip('0') or '0'

    scale = NUMBER_SCALE_DICT[str(index)][0]

    if index == 0:
      if len(number_str) == 3:
        prefixes, converted_text = convert_to_text(number_str, index, number_len)
        suffix = 'अधिक' if number_len > 3 else ""
        combined_prefixes = [combine_morphemes(prefix, NUMBER_SCALE_DICT[str(index)][0]) for prefix in prefixes] or [scale]
        converted_text = combine(converted_text,
                         [combine_morphemes(combined_prefix, suffix) for combined_prefix in combined_prefixes],
                         "")
      else:
        converted_text = convert_to_text(number_str, index, number_len)
    else:
        prefixes = convert_to_text(number_str, index, number_len) if number_str!='1' else []
        suffix = 'अधिक' if texts else ""
        combined_prefixes = [combine_morphemes(prefix, scale) for prefix in prefixes] or [scale]
        converted_text = [combine_morphemes(combined_prefix, suffix) for combined_prefix in combined_prefixes]
    return combine(converted_text, texts, "")
