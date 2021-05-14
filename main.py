import numpy as np
import math
import fractions


def sign(n):
    if n < 0:
        return -1
    else:
        return 1


def QR_decomposition_Householder(A, verbose=False):
    # P = I - 1/beta*u*u^T
    P = []
    i = 0
    for i in range(0, len(A) - 1):
        text = ""
        print(f"Pasul {i + 1}:")
        for a in np.squeeze(np.asarray(A[i:, i])):
            text += f"+({round(a,2)})²"
        sigma = sum([a * a for a in np.squeeze(np.asarray(A[i:, i]))])
        print("Sigma: "+text+" = ", sigma)
        k = -sign(A[i, i]) * np.sqrt(sigma)
        print(f"K: -sign(A[{i}][{i}]={round(A[i,i],2)})*√{sigma} = ", k)
        beta = sigma - k * A[i, i]
        print(f"Beta: σ - (k*A[{i}][{i}]={round(A[i,i],2)})  =", beta)

        u = A[:, i].copy()
        for m in range(0, i):
            u[m] = 0
        u[i] -= k
        u = np.matrix(u)

        print("U:\n", u)
        print("UT:\n", u.T)
        # nu stiu dude de ce pt i = 0 da asa
        if i == 0:
            u_u_t = u.T @ u
        else:
            u_u_t = u @ u.T
        print("U * U^T:\n", u_u_t)
        P.append(np.identity(n=len(A)) - (1 / beta) * u_u_t)
        print(f"P[{i + 1}]:\n", P[i])
        PIxA = P[i] @ A
        print(f"P[{i + 1}] @ A:\n{np.matrix.round(PIxA, 3)}")
        A = PIxA.copy()
        R = PIxA.copy()

    Q = np.identity(n=len(A))
    for p in P:
        Q = Q @ p

    Q = np.squeeze(np.asarray(Q))
    print("MATRICEA Q (Forma fractie): ")
    Q = Q.astype(str)
    for i in range(len(Q)):
        for j in range(len(Q[0])):
            Q[i, j] = fractions.Fraction(Q[i, j]).limit_denominator()
    print(Q)
    print("MATRICE R: ")
    print(np.matrix.round(R, 3))

    return R, Q


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

    # QR decomposition With Householder

    # R, Q = QR_decomposition_Householder(np.array(A), False)  # Turn this to true for all steps
    #
    # print("R: ")
    # print(R)
    #
    # print("Q: ")
    # print(Q)

