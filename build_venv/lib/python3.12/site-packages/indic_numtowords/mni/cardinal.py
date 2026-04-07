from indic_numtowords.mni.data.nums import DIRECT_DICT
from indic_numtowords.mni.data.nums import HUNDREDS_DICT
from indic_numtowords.mni.data.nums import NUMBER_SCALE_DICT
from indic_numtowords.mni.data.nums import VARIATIONS_DICT
from indic_numtowords.mni.utils import combine

def convert_to_text(number_str: str, index: int, prev: bool) -> list[str]:
    """
    Convert a number to its text representation.

    Args:
        number_str (str): The number to convert.

    Returns:
        list[str]: The text representation of the number.
    """
    hundreds = []
    ones = []

    # Handle hundreds place
    if len(number_str) == 3 and number_str.lstrip('0'):
        hundreds.append(HUNDREDS_DICT[number_str[0]][0])

        prefix = number_str[:1]
        number_str = number_str[1:].lstrip('0')

        if number_str:
          prefix_char = 'ꯀ ' if prefix in ['6', '7'] else 'ꯒ '
          ones.append(prefix_char + DIRECT_DICT.get(number_str, [])[0])
          ones.extend(prefix_char + variation for variation in VARIATIONS_DICT.get(number_str, []))

    # Handle tens and ones place
    elif len(number_str) < 3:
      if index > 0 and prev:
        suffix = ' ꯀ' if number_str[-1] in ['6', '7'] else ' ꯒ'
        ones.append(DIRECT_DICT.get(number_str, [])[0] + suffix)
        ones.extend(variation + suffix for variation in VARIATIONS_DICT.get(number_str, []))
      else:
        ones.append(DIRECT_DICT.get(number_str, [])[0])
        ones.extend(variation for variation in VARIATIONS_DICT.get(number_str, []))

    return combine(hundreds, ones)

def process_text(number_str: str, texts: list[str], index: int, number_len: int) -> list[str]:
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

    if number_str in {'00', '000'}:
        return texts

    prev = True if texts else False

    if index == 0 and number_str == '0':
        return combine(convert_to_text(number_str, index, prev), texts)

    converted_text = convert_to_text(number_str.lstrip('0'), index, prev) if index == 0 else combine(NUMBER_SCALE_DICT[str(index)], convert_to_text(number_str.lstrip('0'), index, prev))

    return combine(converted_text, texts)

