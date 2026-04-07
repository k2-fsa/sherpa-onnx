from indic_numtowords.mal.data.nums import exceptions_dict
from indic_numtowords.mal.data.nums import direct_dict
from indic_numtowords.mal.data.nums import joined_dict
from indic_numtowords.mal.data.nums import hundreds_dict

from indic_numtowords.mal.utils import combine


def convert(num):
    num_str = str(num).strip()
    if num_str in exceptions_dict:
        return exceptions_dict[num_str]
    num_str = num_str.lstrip('0')
    n = len(num_str)
    final_word_list = []
    word_list = [""]

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
        if temp_num == '1':
            lis1 = ['ഒരു']
        else:
            lis1 = direct_dict[temp_num]
        num_str = num_str[len(temp_num):]
        num_str = num_str.lstrip('0')
        if num_str == '':
            #then only crores case
            lis2 = ['കോടി']
        else:
            lis2 = ['കോടി']
        inter_list = combine(lis1, lis2)
        word_list = combine(word_list, inter_list)
        n = len(num_str)

    if n == 7 or n == 6:
        #lakh case
        temp_num = num_str[:-5]
        if temp_num == '1':
            lis1 = ['ഒരു']
        else:
            lis1 = direct_dict[temp_num]
        num_str = num_str[len(temp_num):]
        num_str = num_str.lstrip('0')
        if num_str == '':
            #then only lakhs case
            lis2 = ['ലക്ഷം']
        else:
            lis2 = ['ലക്ഷത്തി']
        inter_list = combine(lis1, lis2)
        word_list = combine(word_list, inter_list)
        n = len(num_str)

    if n == 5 or n == 4:
        #thousands case
        temp_num = num_str[:-3]
        lis1 = direct_dict[temp_num]
        num_str = num_str[len(temp_num):]
        num_str = num_str.lstrip('0')
        if num_str == '':
            #then only thousands case
            if temp_num == '1':
                inter_list = ['ആയിരം']
            elif temp_num == '5':
                inter_list = ['അയ്യായിരം']
            else:
                lis1 = [temp[:-1] for temp in lis1]
                lis2 = ['ായിരം']
                inter_list = combine(lis1, lis2, seperator = "")
        else:
            if temp_num == '1':
                inter_list = ['ആയിരത്തി']
            elif temp_num == '5':
                inter_list = ['അയ്യായിരത്തി']
            else:
                lis1 = [temp[:-1] for temp in lis1]
                lis2 = ['ായിരത്തി'] 
                inter_list = combine(lis1, lis2, seperator = "")
        word_list = combine(word_list, inter_list)
        n = len(num_str)

    if n == 3:
        #hundreds case
        if num_str in exceptions_dict:
            inter_list = exceptions_dict[num_str]
            word_list = combine(word_list, inter_list)
            num_str = ""
            n = len(num_str)
        else:
            temp_num = num_str[0]
            inter_list = hundreds_dict[temp_num]
            num_str = num_str[1:]
            if num_str[0] == '5' and temp_num != '9':
                lis1 = [temp[:-1] for temp in inter_list]
                lis2 = joined_dict[num_str]
                inter_list = combine(lis1, lis2, seperator="")
                num_str = num_str[2:]
            word_list = combine(word_list, inter_list)
            num_str = num_str.lstrip('0')
            n = len(num_str)
    
    if n == 2 or n == 1:
        #tens case
        temp_str = direct_dict[num_str]
        word_list = combine(word_list, temp_str)

    final_word_list = word_list + final_word_list
    return [l.strip() for l in final_word_list]
