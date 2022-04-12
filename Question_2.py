from library import mlcg
import numpy as np
from math import sqrt


def integrand(x):
    return sqrt(1 - x**2)


# Assume circle with unit radius
def circle(x, y):
    return x**2 + y**2 - 1


"""
Monte Carlo integration
"""


def piIntegration(func, seed, a, m, N):
    # Generate list of N random points between lims
    xrand = mlcg(seed, a, m, N)
    xrand = np.array(xrand) / m

    summation = 0
    for i in range(N):
        summation += func(xrand[i])

    total = 1 / float(N) * summation

    return total


def piHits(seed1, seed2, a, m, N):
    xrand = mlcg(seed1, a, m, N)
    yrand = mlcg(seed2, a, m, N)
    xrand = np.array(xrand) / m
    yrand = np.array(yrand) / m

    hits = 0
    for i in range(N):
        if circle(xrand[i], yrand[i]) <= 0:
            hits += 1

    estim = hits / N
    return estim


integrated_pi1 = piIntegration(integrand, 234.34, 65, 1021, 100000)
hit_pi1 = piHits(75.345, 36.232, 65, 1021, 100000)
integrated_pi2 = piIntegration(integrand, 234.34, 572, 16381, 100000)
hit_pi2 = piHits(75.345, 36.232, 572, 16381, 100000)
print("(i) piHits = {}, piIntegrated = {}".format(4 * integrated_pi1, 4 * hit_pi1))
print("(ii) piHits = {}, piIntegrated = {}".format(4 * integrated_pi2, 4 * hit_pi2))

# ---------------------------* OUTPUT *--------------------------------------
# (i) piHits = 3.143601478643051, piIntegrated = 3.14048
# (ii) piHits = 3.1411842545237674, piIntegrated = 3.14

# ---------------------------* Comment *--------------------------------------------
# on the value of a and m on pi accuracy
# The choice of seed decides the accuracy of pi that is obtained.
# Here, we notice that for higher 'a' and 'm' value the accuracy shows no particular trend with piHits showing an increased accuracy but piIntegrated shows a decreased accuracy.
