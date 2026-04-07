from indic_numtowords.mai.data.nums import DIRECT_DICT
from indic_numtowords.mai.data.nums import NUMBER_SCALE_DICT
from indic_numtowords.mai.data.nums import VARIATIONS_DICT
from indic_numtowords.mai.utils import combine

def convert_to_text(number_str: str) -> list[str]:
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
        hundreds.append(" ".join(DIRECT_DICT[number_str[0]] + NUMBER_SCALE_DICT['0']))

        if number_str[0] in VARIATIONS_DICT:
            hundreds.extend([" ".join([variation, NUMBER_SCALE_DICT['0'][0]]) for variation in VARIATIONS_DICT[number_str[0]]])

        number_str = number_str[1:].lstrip('0')

    # Handle tens and ones place
    if number_str:
        ones.append(DIRECT_DICT.get(number_str, [])[0])
        ones.extend(VARIATIONS_DICT[number_str]) if number_str in VARIATIONS_DICT else None

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

    if index == 0 and number_str == '0':
        return combine(convert_to_text(number_str), texts)

    converted_text = convert_to_text(number_str.lstrip('0')) if index == 0 else combine(convert_to_text(number_str.lstrip('0')), NUMBER_SCALE_DICT[str(index)])

    return combine(converted_text, texts)