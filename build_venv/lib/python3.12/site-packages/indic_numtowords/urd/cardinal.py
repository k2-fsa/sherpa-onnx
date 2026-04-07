from indic_numtowords.urd.data.nums import exceptions_dict
from indic_numtowords.urd.data.nums import direct_dict
from indic_numtowords.urd.data.nums import higher_dict
from indic_numtowords.urd.data.nums import variations_dict

from indic_numtowords.urd.utils import combine

def convert(num):
    num_str = str(num).strip()
    if num_str in exceptions_dict:
        return exceptions_dict[num_str]
    num_str = num_str.lstrip('0')
    n = len(num_str)
    word_list = [""]
    final_word_list = []

    if n > 9:
        #output individually
        for i in num_str:
            new_list = direct_dict[i]
            word_list = combine(word_list, new_list)
        final_word_list.extend(word_list)
        return final_word_list

    if n == 9 or n == 8:
        #crore case
        temp_num = num_str[:-7]
        lis1 = direct_dict[temp_num]
        lis2 = higher_dict[7]
        inter_list = combine(lis1, lis2)
        word_list = combine(word_list, inter_list)
        num_str = num_str[len(temp_num):]
        num_str = num_str.lstrip('0')
        n = len(num_str)

    if n == 7 or n == 6:
        #lakh case
        temp_num = num_str[:-5]
        lis1 = direct_dict[temp_num]
        lis2 = higher_dict[5]
        inter_list = combine(lis1, lis2)
        word_list = combine(word_list, inter_list)
        num_str = num_str[len(temp_num):]
        num_str = num_str.lstrip('0')
        n = len(num_str)

    if n == 5 or n == 4:
        #thousands case
        temp_num = num_str[:-3]
        lis1 = direct_dict[temp_num]
        lis2 = higher_dict[3]
        inter_list = combine(lis1, lis2)
        word_list = combine(word_list, inter_list)
        num_str = num_str[len(temp_num):]
        num_str = num_str.lstrip('0')
        n = len(num_str)

    if n == 3:
        #hundreds case
        temp_num = num_str[0]
        lis1 = direct_dict[temp_num]
        lis2 = higher_dict[2]
        inter_list = combine(lis1, lis2)
        if num_str in variations_dict:
            final_word_list.extend(combine(word_list, variations_dict[num_str]))
        word_list = combine(word_list, inter_list)
        num_str = num_str[1:]
        num_str = num_str.lstrip('0')
        n = len(num_str)
    
    if n == 2 or n == 1:
        #tens case
        lis2 = direct_dict[num_str]
        lis1 = word_list
        word_list = combine(lis1, lis2)
    
    final_word_list.extend(word_list)
    return [l.strip() for l in final_word_list]
