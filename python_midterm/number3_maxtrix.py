"""
Number 3 answer

"""

import numpy as np

def default_matrinx_matmul(A, B ):
    """
    Matrix multiplication from default python function
    Raise:
        If matrix shape are wrong for matrix multiplication
    """
    result_row = len(A)
    col_of_a = len(A[0])
    row_of_b = len(B)
    result_column = len(B[0])
    result_list = []

    if (col_of_a == row_of_b):
        for _r in range(int(result_row)):
            result_col = []
            for _c in range(result_column):
                result_col.append(0)
            result_list.append(result_col)
    else:
        raise Exception("Can not matrix multiplication")

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result_list[i][j] += A[i][k] * B[k][j]

    print("Python default matrix multiplication  ----> ")
    print(result_list)

def numpy_matrinx_matmul(A, B ):
    """
    Matrix multiplication using numpy

    """
    a = np.array(A)
    b = np.array(B)
    r = np.matmul(a,b)
    print("Numpy matrix multiplication ----> ")
    print(r)



if __name__ == "__main__":
    """
    Matrix multiplication from default python and numpy
    """

    A = [[12, 7, 3,1],
         [4, 5, 6,1],
         [7, 8, 9, 1]]

    B = [[1,2]
        , [2,3]
        , [3,4]
         ,[1,2]]

    default_matrinx_matmul(A, B)
    numpy_matrinx_matmul(A,B)
