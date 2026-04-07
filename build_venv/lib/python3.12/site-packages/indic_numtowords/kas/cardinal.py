from indic_numtowords.kas.data.nums import DIRECT_DICT
from indic_numtowords.kas.data.nums import NUMBER_SCALE_DICT
from indic_numtowords.kas.data.nums import VARIATIONS_DICT
from indic_numtowords.kas.utils import combine

def convert_to_text(number_str: str) -> list[str]:
    """
    Convert a number to its text representation.

    Args:
        number_str (str): The number to convert.

    Returns:
        list[str]: The text representation of the number.
    """
    hundreds_texts = []
    ones_texts = []

    if len(number_str) == 3 and number_str.lstrip('0'):
        if number_str in DIRECT_DICT.keys():
          hundreds_texts.append(DIRECT_DICT[number_str][0])

        scale_key = NUMBER_SCALE_DICT['0'][1] if number_str[0] in ['8', '9'] else NUMBER_SCALE_DICT['0'][0]
        hundreds_texts.insert(0, " ".join(DIRECT_DICT[number_str[0]] + [scale_key]))

        if number_str[0] in VARIATIONS_DICT:
          for variation in VARIATIONS_DICT[number_str[0]]:
              hundreds_texts.insert(0, f"{variation} {scale_key}")

        number_str = number_str[1:].lstrip('0')

        if number_str:
            ones_texts.insert(0, 'تہٕ ' + DIRECT_DICT.get(number_str, [])[0])

            if number_str in VARIATIONS_DICT:
              for item in VARIATIONS_DICT[number_str]:
                ones_texts.insert(0, 'تہٕ ' + item)

    elif len(number_str) in [1, 2]:
        ones_texts.insert(0, DIRECT_DICT.get(number_str, [])[0])

        if number_str in VARIATIONS_DICT:
          for item in VARIATIONS_DICT[number_str]:
            ones_texts.insert(0, item)

    return combine(ones_texts, hundreds_texts)

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

    if index == 0 and number_str == '0':
        return combine(convert_to_text(number_str), texts)

    reversed_scale = NUMBER_SCALE_DICT[str(index)][::-1]

    if index >= 1 and texts:
      if not any('تہٕ' in i for i in texts):
          reversed_scale = combine(['تہٕ'], reversed_scale)

    converted_text = (
        convert_to_text(number_str.lstrip('0')) if index == 0
        else combine(reversed_scale, convert_to_text(number_str.lstrip('0')))
    )
    return combine(texts, converted_text)
