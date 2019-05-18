# import sys
#
#
#
# def number_search():
#     f = open('data.txt', 'r')
#     input_value = input("name")
#     for l in f:
#         if input_value in l:
#             return l.split()[1]
#     f.close()
#
# print(number_search())


def region_book():
    f = open('data.txt', 'r')
    area_number = input("area")
    result = []
    for l in f:
        if area_number in l:
            result.append([l.split()[0], l.split()[1]])

    f.close()
    return result

print(region_book())
