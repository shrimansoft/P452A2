#!/usr/bin/python
# -*- coding: utf-8 -*-
import lib
import math
from matplotlib import pyplot as plt

# -----------------* random seed and other parameters *-------------------

Xo = 1

# (ii)

m2 = 16381
a2 = 572
c = 0
N = 100000

# -----------------* To store random numbers *--------------------------

randomNums = [0] * N
randomNums2 = lib.mlcg(
    Xo,
    m2,
    a2,
    c,
    randomNums,
    N,
)
randomNums2 = [number / (m2 - 1) for number in randomNums2]


def func(x):
    f = (2 * math.sqrt(1 - x**2)) ** 2
    return f


a = -1
b = 1
ss2 = lib.montecarlo_ran_nums(randomNums2, N, func, a, b)
print("Volume of Steinmetz solid by MC method:", "\n For a=572, m=16381 : ", ss2[0])


# ------------* OUTPUT *---------------------
# Volume of Steinmetz solid by MC method:
#  For a=572, m=16381 :  5.333646576997727
