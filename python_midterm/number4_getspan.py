"""
Number 4 answer

"""
def get_span(P):
    """
    Get span
    """
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



if __name__ == "__main__":
    """
    Main 
    P is list contain sales amount.
    print span
    """

    P = [5800, 6000, 6100, 5900, 6050, 4900, 7800, 78001, 78001, 1, 2, 3, 4, 5]
    print(get_span(P))
