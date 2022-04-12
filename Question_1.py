from library import read_csv, lu_decomposition
import numpy as np
import matplotlib.pyplot as plt


def polyLeastSquare(xvals: np.array, yvals: np.array, degree: int = 1):
    """r
    xvals, yvals: data points given as a list (separately) as input
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
                total += xvals[k] ** (i + j)

            A[i, j] = total

    for i in range(params):
        total = 0
        for k in range(n):
            total += xvals[k] ** i * yvals[k]

        b[i] = total

    paramsVec = lu_decomposition(A, b)
    return paramsVec, A


def chebyshev(x: float, order: int) -> float:
    if order == 0:
        return 1
    elif order == 1:
        return 2 * x - 1
    elif order == 2:
        return 8 * x**2 - 8 * x + 1
    elif order == 3:
        return 32 * x**3 - 48 * x**2 + 18 * x - 1


def chebyfit(xvals: np.array, yvals: np.array, degree: int):
    n = len(xvals)
    params = degree + 1
    A = np.zeros((params, params))
    b = np.zeros(params)

    for i in range(params):
        for j in range(params):
            total = 0
            for k in range(n):
                total += chebyshev(xvals[k], j) * chebyshev(xvals[k], i)

            A[i, j] = total

    for i in range(params):
        total = 0
        for k in range(n):
            total += chebyshev(xvals[k], i) * yvals[k]

        b[i] = total

    paramsVec = lu_decomposition(A, b)
    return paramsVec, A


file = read_csv("assign2fit.txt")
xvals = [sub[0] for sub in file]
yvals = [sub[1] for sub in file]

xvals = list(map(float, xvals))
yvals = list(map(float, yvals))

# Condition number of matrix = 21980.9
params, mat1 = polyLeastSquare(xvals, yvals, 3)

# Condition number of matrix = 4.79553
chebyparams, mat2 = chebyfit(xvals, yvals, 3)

a0, a1, a2, a3 = params[0], params[1], params[2], params[3]
c0, c1, c2, c3 = chebyparams[0], chebyparams[1], chebyparams[2], chebyparams[3]
print("Ordered coefficients in original basis: {}".format(params))
print("Ordered coefficients in modified Chebyshev basis: {}".format(chebyparams))

x = np.linspace(0, 1, 100)
y = a0 + a1 * x + a2 * x**2 + a3 * x**3
plt.scatter(xvals, yvals, s=5, label="Datapoints")
plt.plot(x, y, "r", label="Line fit")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Cubic-fit polynomial")
plt.legend()
plt.savefig("q1_plot.png")


# -------------------* OUTPUT *------------------------------
# Ordered coefficients in original basis: [0.5746586674195995, 4.725861442142078, -11.128217777643616, 7.6686776229096685]
# Ordered coefficients in modified Chebyshev basis: [1.1609694790335525, 0.39351446798815237, 0.04684983209010658, 0.23964617571596986]

# For the original basis, the condition number obtained for the corresponding matrix is 21980.9 compared to the condition number in the case of the modified Chebyshev functions where it was found to be 4.79553.

# A lower condition number shows more robustness towards fluctuations/perturbations in the datapoints and thus, the modified Chebyshev basis is a better basis for fitting the given datapoints.
