import numpy as np
import math


def QR_decomposition_Givens(A, verbose=False, eps=1e-7):
    Q = np.identity(len(A))

    for r in range(len(A) - 1):
        if verbose:
            print('Step ' + str(r+1) + ":")

        for i in range(r+1, len(A)):
            f = math.sqrt(A[r][r]**2 + A[i][r]**2)

            if f < eps:
                c = 1
                s = 0
            else:
                c = A[r][r]/f
                s = A[i][r]/f
            for j in range(r+1, len(A)):
                temp = A[r][j]
                A[r][j] = c*A[r][j] + s*A[i][j]
                A[i][j] = -s*temp + c * A[i][j]

            A[i][r] = 0
            A[r][r] = f

            if verbose:
                print("R" + str(r+1) + str(i+1) + "*A:")
                print(A)
                print("")

            for j in range(len(A)):
                temp = Q[r][j]
                Q[r][j] = c*Q[r][j] + s*Q[i][j]
                Q[i][j] = -s*temp + c*Q[i][j]

    return A, Q.T


if __name__ == '__main__':

    with open("a.txt") as f:
        A = [[float(n) for n in line.strip().split(' ')] for line in f]
    with open("b.txt") as f:
        b = [float(n) for n in f.readline().strip().split(' ')]



    #QR decomposition With Givens

    R, Q = QR_decomposition_Givens(np.array(A), False) #Turn this to true for all steps

    print("R: ")
    print(R)

    print("Q: ")
    print(Q)

