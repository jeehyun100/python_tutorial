#
# Python Homework -2-
# Class : Python프로그래밍(GEV6103-01)
# Created by : 김후빈, 백지현, 지동근
# ==============================================================================

def example_basic_1():
    data = [2.3, 3.5, -5.7, 6.9, -4.2, 8.2, 1.1, -1.5, 3.8, 7.2]
    average = sum(data) / len(data)
    return "Example_basic_1 average output ==> ", round(average, 2), data


def example_basic_2(input_param=None):
    if input_param is None:
        n = int(input("Enter number: "))
    else:
        n = input_param
    rev = 0
    while n > 0:
        dig = n % 10
        rev = rev * 10 + dig
        n = n // 10
    return "Example_basic_2 output ==> ", rev, n


def check_integer_input_values(input_text="Enter number: ", negative=True):
    while True:
        try:
            n = int(input(input_text))
            if negative is True:
                break
            else:
                if n > 0:
                    break
                else:
                    print("Validation Error : Input positive integer number")
        except ValueError:
            print("Validation Error : Input integer number")
    return n


def example_basic_3(input_param=None):
    # Find the number of digits in a number.
    if input_param is None:
        n = check_integer_input_values()
    else:
        n = input_param
    result = len(str(abs(n)))
    return "Find the number of digits output ==> ", result, n


def example_basic_4(input_param=None):
    # Check if a number is a palindrome.
    if input_param is None:
        n = check_integer_input_values()
    else:
        n = input_param
    temp = n
    rev = 0
    while n > 0:
        dig = n % 10
        rev = rev * 10 + dig
        n = n // 10
    if temp == rev:
        result = "{0} is a palindrome.".format(temp)
    else:
        result = "{0} is not a palindrome.".format(temp)
    return "Palindrome check output ==> ", result


def example_lists_1():
    # Find the largest and the second largest number in a list

    import random

    # Make random number of list
    random_value_list = [int(1000*random.random()) for _ in range(10)]
    length = len(random_value_list)
    # Sort a list
    list_temp = list(random_value_list)
    list_temp.sort()
    largest = list_temp[length-1]
    second_largest = list_temp[length-2]
    return "The Largest and 2nd Output ==>", largest, second_largest, random_value_list


def example_lists_2():
    # Merge two lists and sort it

    import random

    # Make random number of lists
    random_value_list1 = [int(1000*random.random()) for _ in range(10)]
    random_value_list2 = [int(1000*random.random()) for _ in range(10)]
    merged_sorted_list = sorted(random_value_list1 + random_value_list2)
    return "Sorted merged list output ==>", merged_sorted_list, "List1+List2=", random_value_list1, random_value_list2


def example_lists_3():
    # Swap the first and the last value of a list

    import random

    # Make random number of list
    random_value_list = [int(1000*random.random()) for _ in range(10)]
    temp_list = list(random_value_list)
    temp_element = temp_list[0]
    temp_list[0] = temp_list[len(temp_list)-1]
    temp_list[len(temp_list) - 1] = temp_element
    return "Swap the first and the last value output ==>", temp_list, "The original list=", random_value_list


def example_lists_4():
    # Remove the duplicate items from a list

    duplated_value_list = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    unique_list = list(set(duplated_value_list))
    return "Remove the duplicate items from a list Output ==>", unique_list,\
           "The original list=", duplated_value_list


def example_lists_5(input_param=None):
    # Read a list of words and return the length of the longest one

    word_list = []
    if input_param is None:
        n = check_integer_input_values()
        for num_of_element in range(n):
            input_word = input("Input element {0}".format(num_of_element) + ":")
            word_list.append(input_word)
    else:
        word_list = input_param
    sorted_list = sorted(word_list, key=len)
    return "The length of the longest one output ==>", sorted_list[-1], "The original list=", word_list


def example_string_1(input_param1=None, input_param2=None):
    # Detect if two strings are anagrams
    if input_param1 is None or input_param2 is None:
        s1 = input("Input the first string:")
        s2 = input("Input the 2nd string:")
    else:
        s1 = input_param1.lower()
        s2 = input_param2.lower()

    if sorted(s1) == sorted(s2):
        result = "The strings are anagrams."
    else:
        result = "The strings aren't anagrams."
    return "The anagrams check output ==>", result, s1, "and", s2,


def example_string_2(input_param=None):
    # Count the number of vowels in a string
    if input_param is None:
        s1 = input("Please type a sentence: ")
    else:
        s1 = input_param
    vowels_cnt = sum(map(s1.lower().count, "aeiou"))
    return "Count the number of vowels output ==>", vowels_cnt, s1


def example_string_3(input_params=None):
    # Calculate the number of upper case letters and lower case letters in a string
    if input_params is None:
        s1 = input("Please type a sentence: ")
    else:
        s1 = input_params
    upper = sum(1 for i in s1 if i.isupper())
    lower = sum(1 for i in s1 if i.islower())
    return "Calculate the upper and lower case output ==>", upper, lower, s1


def example_dictionary_1():
    # Concatenate two dictionaries into one

    import random

    d1 = {int(1000 * random.random()): k for k in range(5)}
    d2 = {int(1000 * random.random()): k for k in range(5)}
    d1.update(d2)
    return "Concatenate two dictionaries output ==>", d1


def example_dictionary_2():
    # Sum all items in a dictionary

    import random

    d1 = {chr(k+65): int(1000 * random.random()) for k in range(5)}
    sum_all_dict = sum(d1.values())
    return "Sum all items in a dictionary output ==>", sum_all_dict, d1


def example_dictionary_3():
    # Map two lists into a dictionary

    length_of_list = 10
    list_key = [chr(i+65) for i in range(length_of_list)]
    list_value = [k for k in range(length_of_list)]
    d1 = dict(zip(list_key, list_value))
    return "Map two lists into a dictionary output==>", d1


def example_dictionary_4(input_param=None):
    # Create a dictionary with key as first character and value as words starting with that character

    split_string = []
    d1 = {}
    if input_param is None:
        n = check_integer_input_values()
        for i in range(n):
            test_string = input("Input string " + str(i) + ":")
            split_string.append(test_string)
    else:
        test_string = input_param
        split_string = test_string.split()

    for word in split_string:
        if word[0] not in d1.keys():
            d1[word[0]] = []
            d1[word[0]].append(word)
        else:
            if word not in d1[word[0]]:
                d1[word[0]].append(word)
    return "Create a dictionary with key as first character output ==>", d1


def example_sets_1(input_param=None):
    # Count the number of vowels present in a string using sets

    if input_param is None:
        s1 = input("Input string :")
    else:
        s1 = input_param
    vowel_sets = set('aeiou')
    result = sum([1 for s in s1.lower() if s in vowel_sets])
    return "Count the number of vowels output ==>", result, s1


def example_sets_2(input_param1=None, input_param2=None):
    # Check common character letters in two input strings

    if input_param1 is None or input_param2 is None:
        s1 = input("Input the first string:")
        s2 = input("Input the 2nd string:")
    else:
        s1 = input_param1.lower()
        s2 = input_param2.lower()
    result = ''.join(set(s1).intersection(s2))
    return "Check common character letters output ==>", result


def recur_sum(recur_list):
    if len(recur_list) == 0:
        return 0
    else:
        return recur_list[0] + recur_sum(recur_list[1:])


def example_recursions_1():
    # Find the sum of elements in a list recursively

    import random

    # Make random number of list
    random_value_list = [int(1000*random.random()) for _ in range(2)]
    result = recur_sum(random_value_list)
    return "Sum of elements in a list recursively output ==>", result, random_value_list


def recur_pow(base, exp):
    if exp == 1:
        return base
    else:
        return base * recur_pow(base, exp-1)


def example_recursions_2(input_param1=None, input_param2=None):
    # Find the power of a number using recursion

    if input_param1 is None or input_param2 is None:
        b1 = check_integer_input_values("Input base :")
        e1 = check_integer_input_values("Input exp :", negative=False)
    else:
        b1 = input_param1
        e1 = input_param2
    result = recur_pow(b1, e1)
    return "The power of a number using recursion ==>", result, b1, e1


def recur_reverse_str(input_string):
    if len(input_string) == 0:
        return input_string
    else:
        return recur_reverse_str(input_string[1:]) + input_string[0]


def example_recursions_3(input_param=None):
    # Reverse a string using recursion

    if input_param is None:
        s1 = input("Input string :")
    else:
        s1 = input_param

    result = recur_reverse_str(s1)
    return "Reverse a string using recursion output ==>", result, s1


if __name__ == '__main__':
    print("example_basic_1 ===> ", example_basic_1())
    print("example_basic_2 ===> ", example_basic_2(324))
    print("example_basic_3 ===> ", example_basic_3(12345))
    print("example_basic_4 ===> ", example_basic_4(12321))
    print("###################################################################################")
    print("example_lists_1 ===> ", example_lists_1())
    print("example_lists_2 ===> ", example_lists_2())
    print("example_lists_3 ===> ", example_lists_3())
    print("example_lists_4 ===> ", example_lists_4())
    print("example_lists_5 ===> ", example_lists_5(['apple', 'pear', 'banana', 'strawberry']))
    print("###################################################################################")
    print("example_string_1 ===> ", example_string_1('Elvis', 'Lives'))
    print("example_string_2 ===> ", example_string_2('Apple'))
    print("example_string_3 ===> ", example_string_3('I am a boy'))
    print("###################################################################################")
    print("example_dictionary_1 ===> ", example_dictionary_1())
    print("example_dictionary_2 ===> ", example_dictionary_2())
    print("example_dictionary_3 ===> ", example_dictionary_3())
    print("example_dictionary_4 ===> ", example_dictionary_4('apple, pear, banana, strawberry'))
    print("###################################################################################")
    print("example_sets_1 ===> ", example_sets_1('Apple'))
    print("example_sets_2 ===> ", example_sets_2('apple', 'bAnAnA'))
    print("###################################################################################")
    print("example_recursions_1 ===> ", example_recursions_1())
    print("example_recursions_2 ===> ", example_recursions_2(2, 5))
    print("example_recursions_3 ===> ", example_recursions_3('Hello world'))
