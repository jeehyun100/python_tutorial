def bubble_sort(alist):
    for i in range(len(alist)):
        for i in range(len(alist)-i-1):
            # 앞의 값이 더 크다 -> 바꿔줘야 한다(오름차순)
            if alist[i] > alist[i+1]:
                temp = alist[i+1]
                alist[i+1] = alist[i]
                alist[i] = temp
    return(alist)

test_list = [11,3,24,55,13,44,2,1]
print(bubble_sort(test_list))
