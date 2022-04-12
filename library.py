import numpy as np
from math import sqrt
import random

"""
Helper Functions
"""
# Read matrix from a file given as a string (space separated file)
def read_matrix(file):
    with open(file, "r") as f:
        a = [[int(num) for num in line.split(" ")] for line in f]

    return a


# Prints matrix as written on paper
def mat_print(a):
    for i in range(len(a)):
        print(a[i])


# Calculates the norm of a vector x
def norm(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2

    return total ** (1 / 2)


# Vector subtraction
def vec_sub(a, b):
    if len(a) != len(b):
        exit()
    else:
        return [x1 - x2 for (x1, x2) in zip(a, b)]


# Matrix multiplication
def matmul(a, b):
    product = [
        [sum(i * j for i, j in zip(a_row, b_col)) for b_col in zip(*b)] for a_row in a
    ]

    return product


# Matrix and vector multiplication
def vecmul(A, b):
    if len(A) == len(b):
        vec = [0 for i in range(len(b))]
        for i in range(len(b)):
            total = 0
            for j in range(len(b)):
                vec[i] += A[i][j] * b[j]
        return vec


# Matrix transpose
def transpose(a):
    tr = [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]
    return tr


# Inner product
def inner_prod(a, b):
    atr = transpose(a)
    result = matmul(atr, b)
    return result


# Vector dot product
def dotprod(a, b):
    if len(a) != len(b):
        exit()
    else:
        total = 0
        for i in range(len(a)):
            total += a[i] * b[i]

        return total


def gs_decompose(A):
    U = [[0 for i in range(len(A))] for j in range(len(A))]
    L = [[0 for i in range(len(A))] for j in range(len(A))]

    for i in range(len(A)):
        for j in range(len(A)):
            if i >= j:
                L[i][j] = A[i][j]
            else:
                U[i][j] = A[i][j]

    return L, U


def read_csv(path):
    with open(path, "r+") as file:
        results = []

        for line in file:
            line = line.rstrip("\n")  # remove `\n` at the end of line
            items = line.split(",")
            results.append(list(items))

        # after for-loop
        return results


##################################################
############ MATRIX INVERSION ####################
##################################################
# Forward-backward substitution function which returns the solution x = [x1, x2, x3, x4]
def forward_backward(U: list, L: list, b: list) -> list:
    y = [0 for i in range(len(b))]

    for i in range(len(b)):
        total = 0
        for j in range(i):
            total += L[i][j] * y[j]
        y[i] = b[i] - total

    x = [0 for i in range(len(b))]

    for i in reversed(range(len(b))):
        total = 0
        for j in range(i + 1, len(b)):
            total += U[i][j] * x[j]
        x[i] = (y[i] - total) / U[i][i]

    return x


"""
Gauss Jordan
"""


def gauss_jordan(A: list, b: list) -> list:
    def partial_pivot(A: list, b: list):
        n = len(A)
        for i in range(n - 1):
            if abs(A[i][i]) < 1e-10:
                for j in range(i + 1, n):
                    if abs(A[j][i]) > abs(A[i][i]):
                        A[j], A[i] = A[i], A[j]  # interchange A[i] and A[j]
                        b[j], b[i] = b[i], b[j]  # interchange b[i] and b[j]

    n = len(A)
    partial_pivot(A, b)
    for i in range(n):
        pivot = A[i][i]
        b[i] = b[i] / pivot
        for c in range(i, n):
            A[i][c] = A[i][c] / pivot

        for k in range(n):
            if k != i and A[k][i] != 0:
                factor = A[k][i]
                b[k] = b[k] - factor * b[i]
                for j in range(i, n):
                    A[k][j] = A[k][j] - factor * A[i][j]

    x = b
    return x


# def gauss_jordan(A: np.ndarray, b: np.ndarray) -> np.ndarray:
#     # Pivots for a given row k
#     def partial_pivot(A: np.ndarray, b: np.ndarray, k: int) -> tuple:
#         n = len(A)
#         if abs(A[k,k]) < 1e-10:
#             for i in range(k+1, n):
#                 if abs(A[i,k]) > abs(A[k,k]):
#                     A[k], A[i] = A[i], A[k]
#                     b[k], b[i] = b[i], b[k]

#         return A, b

#     n = len(A)
#     for i in range(n):
#         A, b = partial_pivot(A, b, i)
#         # set pivot row
#         pivot = A[i,i]
#         # Divide row with pivot (and corresponding operation on b)
#         for j in range(i, n):
#             A[i,j] /= pivot

#         b[i] /= pivot

#         for j in range(n):
#             if abs(A[j,i]) > 1e-10 and j != i:
#                 temp = A[j,i]
#                 for k in range(i, n):
#                     A[j,k] = A[j,k] - temp * A[i,k]
#                 b[j] = b[j] - temp * b[i]

#     return b


"""
LU Decomposition
"""


def lu_decomposition(A: list, b: list) -> list:
    # Partial pivoting with matrix 'A', vector 'b'
    def partial_pivot(A: list, b: list):
        count = 0  # keeps a track of number of exchanges
        n = len(A)
        for i in range(n - 1):
            if abs(A[i][i]) < 1e-10:
                for j in range(i + 1, n):
                    if abs(A[j][i]) > abs(A[i][i]):
                        A[j], A[i] = (
                            A[i],
                            A[j],
                        )  # interchange ith and jth rows of matrix 'A'
                        count += 1
                        b[j], b[i] = (
                            b[i],
                            b[j],
                        )  # interchange ith and jth elements of vector 'b'

        return A, b, count

    # Crout's method of LU decomposition
    def crout(A: list):
        U = [[0 for i in range(len(A))] for j in range(len(A))]
        L = [[0 for i in range(len(A))] for j in range(len(A))]

        for i in range(len(A)):
            L[i][i] = 1

        for j in range(len(A)):
            for i in range(len(A)):
                total = 0
                for k in range(i):
                    total += L[i][k] * U[k][j]

                if i == j:
                    U[i][j] = A[i][j] - total

                elif i > j:
                    L[i][j] = (A[i][j] - total) / U[j][j]

                else:
                    U[i][j] = A[i][j] - total

        return U, L

    partial_pivot(A, b)
    U, L = crout(A)
    x = forward_backward(U, L, b)
    return x


"""
Cholesky Decomposition
"""


def cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(A)
    L = np.zeros((n, n))

    for i in range(n):
        total1 = 0
        for k in range(i):
            total1 += L[i, k] ** 2

        L[i, i] = np.sqrt(A[i, i] - total1)

        for j in range(i + 1, n):
            total2 = 0
            for k in range(i):
                total2 += L[i, k] * L[j, k]

            L[j, i] = 1 / L[i, i] * (A[i, j] - total2)

    x = forward_backward(L.T, L, b)
    return x


"""
Jacobi Method
"""


def jacobi(A: list, b: list, tol: float) -> list:
    n = len(A)
    x = [1 for i in range(n)]  # define a dummy vector for storing solution vector
    xold = [0 for i in range(n)]
    iterations = []
    residue = []
    count = 0
    while norm(vec_sub(xold, x)) > tol:
        iterations.append(count)
        count += 1
        residue.append(norm(vec_sub(xold, x)))
        xold = x.copy()
        for i in range(n):
            total = 0
            for j in range(n):
                if i != j:
                    total += A[i][j] * x[j]

            x[i] = 1 / A[i][i] * (b[i] - total)

    return x, iterations, residue


"""
Gauss-Seidel
"""


def gauss_seidel(A: list, b: list, tol: float) -> list:
    n = len(A)
    x = [0 for i in range(n)]
    xold = [1 for i in range(n)]
    iterations = []
    residue = []
    count = 0

    while norm(vec_sub(x, xold)) > tol:
        xold = x.copy()
        iterations.append(count)
        count += 1
        for i in range(n):
            d = b[i]
            for j in range(n):
                if j != i:
                    d -= A[i][j] * x[j]

            x[i] = d / A[i][i]

        residue.append(norm(vec_sub(x, xold)))

    return x, iterations, residue


"""
Conjugate Gradient
"""


def conjgrad(A: list, b: list, tol: float) -> list:
    """r
    Function to solve a set of linear equations using conjugate gradient
    method. However, this works strictly for symmetric and positive definite
    matrices only.
    """
    n = len(b)
    x = [1 for i in range(n)]
    r = vec_sub(b, vecmul(A, x))
    d = r.copy()
    rprevdot = dotprod(r, r)
    iterations = []
    residue = []
    count = 0  # counts the number of iterations

    # convergence in n steps
    for i in range(n):
        iterations.append(count)
        Ad = vecmul(A, d)
        alpha = rprevdot / dotprod(d, Ad)
        for j in range(n):
            x[j] += alpha * d[j]
            r[j] -= alpha * Ad[j]
        rnextdot = dotprod(r, r)
        residue.append(sqrt(rnextdot))
        count += 1

        if sqrt(rnextdot) < tol:
            return x, iterations, residue

        else:
            beta = rnextdot / rprevdot
            for j in range(n):
                d[j] = r[j] + beta * d[j]
            rprevdot = rnextdot


##################################################
########### EIGENVALUE PROBLEM ###################
##################################################

"""
Power Method
"""


def power_method(A: np.ndarray, x: np.ndarray, tol: float, eignum: int = 1) -> tuple:
    """r
    Function to evaluate the eigenvalues and corresponding eigenvectors for a
    given matrix `A`, a random vector `x` of same dimension, a tolerance `tol`
    and number of eigenvalues `eignum` (either 1 or 2).
    """
    n = len(A)
    x = x / np.linalg.norm(x)
    y = x.copy()
    if eignum == 1:
        diff = 1
        while diff > tol:
            xnew = A @ x
            eigval = np.dot(xnew, x) / np.dot(x, x)
            xnew = xnew / np.linalg.norm(xnew)
            diff = np.linalg.norm(xnew - x)
            x = xnew.copy()

        vec = xnew

        return eigval, vec

    elif eignum == 2:
        diff = 1
        while diff > tol:
            xnew = A @ x
            eigval1 = np.dot(xnew, x) / np.dot(x, x)
            xnew = xnew / np.linalg.norm(xnew)
            diff = np.linalg.norm(xnew - x)
            x = xnew.copy()

        vec1 = xnew

        A = A - eigval1 * np.outer(vec1, vec1.T)
        diff = 1
        while diff > tol:
            ynew = A @ y
            eigval2 = np.dot(ynew, y) / np.dot(y, y)
            ynew = ynew / np.linalg.norm(ynew)
            diff = np.linalg.norm(ynew - y)
            y = ynew.copy()

        vec2 = ynew

        return eigval1, eigval2, vec1, vec2


"""
Jacobi Method (using Given's rotation)
"""


def given_jacobi(A: np.ndarray, tol: float) -> tuple:
    """r
    Generates a transformation matrix to kill the non-zero off-diagonal
    elements and diagonalize the original matrix to find the eigenvalues and
    eigenvectors
    """

    def maxElement(A: np.ndarray) -> tuple:
        """r
        To find the largest off-diagonal element in the matrix
        """
        n = len(A)
        amax = 0.0
        for i in range(n):
            for j in range(n):
                if (i != j) and (abs(A[i, j]) >= amax):
                    amax = abs(A[i][j])
                    k = i
                    l = j

        return amax, k, l

    def givensRotation(A: np.ndarray, S: np.ndarray, k: int, l: int):
        n = len(A)
        diffA = A[l, l] - A[k, k]
        if abs(A[k][l]) < abs(diffA) * 1e-20:
            t = A[k, l] / diffA

        else:
            psi = diffA / (2.0 * A[k][l])
            t = 1.0 / (abs(psi) + np.sqrt(psi**2 + 1.0))
            if psi < 0.0:
                t = -t

        c = 1.0 / np.sqrt(t**2 + 1.0)
        s = t * c
        tau = s / (1.0 + c)
        temp = A[k, l]
        A[k, l] = 0.0
        A[k, k] = A[k, k] - t * temp
        A[l, l] = A[l, l] + t * temp

        for i in range(k):
            temp = A[i, k]
            A[i, k] = temp - s * (A[i, l] + tau * temp)
            A[i, l] = A[i, l] + s * (temp - tau * A[i, l])

        for i in range(k + 1, l):
            temp = A[k, i]
            A[k, i] = temp - s * (A[i, l] + tau * A[k, i])
            A[i, l] = A[i, l] + s * (temp - tau * A[i, l])

        for i in range(l + 1, n):
            temp = A[k, i]
            A[k, i] = temp - s * (A[l, i] + tau * temp)
            A[l, i] = A[l, i] + s * (temp - tau * A[l, i])

        for i in range(n):
            temp = S[i, k]
            S[i, k] = temp - s * (S[i, l] + tau * S[i, k])
            S[i, l] = S[i, l] + s * (temp - tau * S[i, l])

    n = len(A)
    maxRot = n**2
    S = np.identity(n)
    for i in range(maxRot):
        amax, k, l = maxElement(A)
        if amax < tol:
            return np.diagonal(A), S

        givensRotation(A, S, k, l)


##################################################
####### STATISTICAL DESCRIPTION OF DATA ##########
##################################################
"""
Jackknife: Finds the mean and variance of population via finite sampling
"""


def jackknife(yis: list) -> tuple:
    delAverages = []  # holds all the yk averages for each j
    n = len(yis)
    for i in range(n):
        total = sum(yis)
        total -= yis[i]
        total = total / (n - 1)
        delAverages.append(total)

    jkAverage = 1 / n * sum(delAverages)  # calculate jackknife average

    for j in range(n):
        err = 0
        err += (delAverages[j] - jkAverage) ** 2

    jkError = err / n  # calculate jackknife standard error

    return jkAverage, jkError


"""
Bootstrap: Resampling of data points from unknown distribution
Implementation of empirical bootstrap
"""


def bootstrap(xis: list, B: int) -> tuple:
    bootSamples = []
    n = len(xis)
    for i in range(B):
        xis_sampled = random.choices(xis, k=n)
        bootSamples.append(xis_sampled)

    for j in range(B):
        xalphaAvg = 1 / n * sum(bootSamples[j])
        xb = 1 / B * sum(xalphaAvg)


"""
Linear Regression
"""
# Returns the intercept and slope for a linear regression (chi square fit),
# given a set of data points
def linear_fit(xvals: np.ndarray, yvals: np.ndarray, variance: np.ndarray):
    """r
    xvals, yvals: data points given as a list (separately) as input
    Return values:
        a: intercept
        b: slope
        delA2: variance of a
        delB2: variance of b
        cov: covariance of a, b
        chi2: chi^2 / dof
        Linear plot: y = a + b*x
    """
    n = len(xvals)  # number of datapoints

    s, sx, sy, sxx, sxy = 0, 0, 0, 0, 0
    for i in range(n):
        s += 1 / variance[i] ** 2
        sx += xvals[i] / variance[i] ** 2
        sy += yvals[i] / variance[i] ** 2
        sxx += xvals[i] ** 2 / variance[i] ** 2
        sxy += xvals[i] * yvals[i] / variance[i] ** 2

    delta = s * sxx - sx**2
    a = (sxx * sy - sx * sxy) / delta
    b = (s * sxy - sx * sy) / delta

    # calculate chi^2 / dof
    dof = n - 2
    chi2 = 0
    for i in range(n):
        chi2 += (yvals[i] - a - b * xvals[i]) ** 2 / variance[i] ** 2

    delA2 = sxx / delta
    delB2 = s / delta
    cov = -sx / delta
    return a, b, delA2, delB2, cov, chi2


"""
Polynomial Fit
"""


def polynomial(
    xvals: np.ndarray, yvals: np.ndarray, variance: np.ndarray, degree: int = 1
):
    """r
    xvals, yvals: data points given as a list (separately) as input
    Variance: given as input for every datapoint
    degree: the degree of polynomial

    Return values:
        a0, a1, a2
        Polynomial plot: y = a0 + a1*x + a2*x**2 + ... + a_{n-1}*x^{n-1}

    Returns a linear fit if `degree` not predefined
    """
    n = len(xvals)
    params = degree + 1  # no. of parameters
    A = np.zeros((params, params))  # Matrix
    b = np.zeros(params)  # Vector

    for i in range(params):
        for j in range(params):
            total = 0
            for k in range(n):
                total += xvals[k] ** (i + j) / variance[k] ** 2

            A[i, j] = total

    for i in range(params):
        total = 0
        for k in range(n):
            total += (xvals[k] ** i * yvals[k]) / variance[k] ** 2

        b[i] = total

    paramsVec = lu_decomposition(A, b)
    return paramsVec


"""
Discrete Fourier Transform
"""


def dft(x: np.ndarray) -> np.ndarray:
    N = len(x)
    n = np.ndarray([i for i in range(N)])
    k = n.T
    e = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(e, x)
    return X


"""
Pseudorandom number generator
"""


def mlcg(seed: float, a: float, m: float, num: int) -> list:
    x = seed
    rands = []
    for i in range(num):
        x = (a * x) % m
        rands.append(x)

    return rands


"""
Monte Carlo integration
"""


def monteCarlo(func, N):
    # Generate list of N random points between lims
    xrand = mlcg(234.34, 65, 1, N)

    summation = 0
    for i in range(N):
        summation += func(xrand[i])

    total = 1 / float(N) * summation

    return total
