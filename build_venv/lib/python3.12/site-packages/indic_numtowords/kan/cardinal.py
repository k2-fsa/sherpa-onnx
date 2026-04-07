from indic_numtowords.kan.data.nums import direct_dict
from indic_numtowords.kan.data.nums import exceptions_dict
from indic_numtowords.kan.data.nums import higher_dict
from indic_numtowords.kan.data.nums import hundreds_dict

def convert(num):
    num_str = str(num).strip()
    if num_str in exceptions_dict:
        return exceptions_dict[num_str]
    num_str = num_str.lstrip('0')
    n = len(num_str)
    word = []
    hundreds_flag = False

    if n > 9:
        #output individually
        for i in num_str:
            if len(word)==0:
                for x in range (0, len(direct_dict[i])):
                    word.append(direct_dict[i][x] + " ")
            else:
                new_word = []
                for x in range (0, len(direct_dict[i])):
                    for y in range(len(word)):
                        new_word.append(word[y] + " " + direct_dict[i][x] + " ")
                word = new_word
        for y in range(len(word)):
            word[y] = word[y].strip()
        return word

    if n == 9 or n == 8:
        #crore case
        temp_num = num_str[:-7]
        temp_str = direct_dict[temp_num]
        if len(word) == 0:
            for x in range (0, len(temp_str)):
                word.append(temp_str[x]+ " " + higher_dict[7] + " ")
            if temp_num=='1':
                word.append(higher_dict[7] + " ")
        else:
            new_word = []
            for x in range (0, len(temp_str)):
                for y in range(len(word)):
                    num = word[y]+temp_str[x] + " " + higher_dict[7] + " "
                    new_word.append(num)
            word = new_word  
        num_str = num_str[len(temp_num):]
        num_str = num_str.lstrip('0')
        n = len(num_str)

    if n == 7 or n == 6:
        #lakh case
        temp_num = num_str[:-5]
        temp_str = direct_dict[temp_num]
        if len(word) == 0:
            for x in range (0, len(temp_str)):
                word.append(temp_str[x]+ " " + higher_dict[5] + " ")
            if temp_num=='1':
                word.append(higher_dict[5] + " ")
        else:
            new_word = []
            for x in range (0, len(temp_str)):
                for y in range(len(word)):
                    num = word[y]+temp_str[x] + " " + higher_dict[5] + " "
                    new_word.append(num)
            word = new_word
        num_str = num_str[len(temp_num):]
        num_str = num_str.lstrip('0')
        n = len(num_str)

    if n == 5 or n == 4:
        #thousands case
        temp_num = num_str[:-3]
        temp_str = direct_dict[temp_num]
        if len(word) == 0:
            for x in range (0, len(temp_str)):
                word.append(temp_str[x]+ " " + higher_dict[3] + " ")
            if temp_num=='1':
                word.append(higher_dict[3] + " ")
        else:
            new_word = []
            for x in range (0, len(temp_str)):
                for y in range(len(word)):
                    num = word[y]+temp_str[x] + " " + higher_dict[3] + " "
                    new_word.append(num)
            word = new_word
        num_str = num_str[len(temp_num):]
        num_str = num_str.lstrip('0')
        n = len(num_str)
        
    if n==3:
        #hundreds case
        if num_str in exceptions_dict:
            new_word = []
            for y in range(0,len(word)):
                x = word[y]
                for z in range (len(exceptions_dict[num_str])):
                    x += exceptions_dict[num_str][z]
                    new_word.append(x)
            word = new_word
        else:
            temp_num = num_str[0]
            temp_str = hundreds_dict[int(temp_num)]
            num_str = num_str[len(temp_num):]
            num_str = num_str.lstrip('0')
            n = len(num_str)
            if num_str!= '':
                temp_str = temp_str[:-1] 
                hundreds_flag = True
            else:
                temp_str = temp_str[:-1] 
                temp_str += 'ರು'
            
            if len(word)==0:
                word.append(temp_str + " ")    
            else:
                for y in range(len(word)):
                    word[y]+= temp_str + " "
                    
    if n == 2 or n == 1:
        #tens case
        temp_str = direct_dict[num_str]
        if hundreds_flag:
            for z in range(len(word)):
                word[z] = word[z][:-1]
            for x in range (0, len(temp_str)):
                for y in range (len(word)):
                    if int(num_str)>10 and int(num_str)<20:
                        word[y] += 'ರ' + temp_str[x] + " "
                    elif int(num_str)>20 and int(num_str)<30:
                        word[y] += 'ರಿ' + temp_str[x][1:] + " "
                    elif int(num_str)>30 and int(num_str)<50:
                        word[y] += 'ರ' + temp_str[x] + " "
                    elif int(num_str)>50 and int(num_str)<60:
                        word[y] += 'ರೈ' + temp_str[x][1:] + " "
                    elif int(num_str)>60 and int(num_str)<70:
                        word[y] += 'ರ' + temp_str[x][1:] + " "
                    elif int(num_str)>70 and int(num_str)<90:
                        word[y] += 'ರೆ' + temp_str[x][1:] + " "
                    elif int(num_str)>90 and int(num_str)<100:
                        word[y] += 'ರ' + temp_str[x] + " "
        else:
            if len(word)==0:
                for x in range (0, len(temp_str)):
                    word.append(temp_str[x])
            else:
                for x in range (0, len(temp_str)):
                    for y in range(len(word)):
                        word[y]+=temp_str[x] + " " 

    for w in range(len(word)):
        word[w] = word[w].strip()
    return word
