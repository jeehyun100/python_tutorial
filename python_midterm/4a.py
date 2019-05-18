P = [5800,6000,6100,5900,6050,4900,7800]



def get_span():
    S = []
    before_value = 0
    idx = 0
    for price in P :
        after_value = price
        if idx == 0 :
            S.append(1)
        else:
            if before_value < after_value:
                S.append(S[idx-1] + 1)
            else:
                S.append(1)

        before_value = after_value
        idx+=1
    return S

print(get_span())
#print(S)
