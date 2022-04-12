#!/usr/bin/python
# -*- coding: utf-8 -*-
import lib
import math
from matplotlib import pyplot as plt

# -----------------* random seed and other parameters *----------------------

Xo = 1

# (i)

m1 = 1021
a1 = 65

# (ii)

m2 = 16381
a2 = 572
c = 0
N = 100000

# ---------------* To store random numbers *-------------------------

randomNums = [0] * N
randomNums1 = lib.mlcg(
    Xo,
    m1,
    a1,
    c,
    randomNums,
    N,
)

# to keep it between 1 and 2

randomNums1 = [number / (m1 - 1) for number in randomNums1]
randomNums2 = lib.mlcg(
    Xo,
    m2,
    a2,
    c,
    randomNums,
    N,
)
randomNums2 = [number / (m2 - 1) for number in randomNums2]

# print(randomNums1)

# (A) Pi by throwing the points Area of circle/square = Points in circle/total Points

count1 = 0
count2 = 0
for i in range(N):
    x1 = randomNums1[i]
    y1 = randomNums1[i - 1]
    x2 = randomNums2[i]
    y2 = randomNums2[i - 1]
    r1 = math.sqrt(x1**2 + y1**2)
    r2 = math.sqrt(x2**2 + y2**2)
    if r1 < 1:
        count1 += 1
    if r2 < 1:
        count2 += 1
pi1 = 4 * count1 / N
pi2 = 4 * count2 / N
print(
    "Value of Pi by throwing the points : \n For a=65, m=1021 : ",
    pi1,
    "\n For a=572, m=16381 : ",
    pi2,
)


# (B) solving the integral given below by Monte Carlo


def func(x):
    f = 4 * math.sqrt(1 - x**2)
    return f


a = 0
b = 1

pi1 = lib.montecarlo_ran_nums(randomNums1, N, func, a, b)
pi2 = lib.montecarlo_ran_nums(randomNums2, N, func, a, b)
print(
    "Value of Pi by solving the integral : \n For a=65, m=1021 : ",
    pi1[0],
    "\n For a=572, m=16381 : ",
    pi2[0],
)

# ------* OUTPUT *----------------

# Value of Pi by throwing the points :
#  For a=65, m=1021 :  3.14512
#  For a=572, m=16381 :  3.1398
# Value of Pi by solving the integral :
#  For a=65, m=1021 :  3.139575266904137
#  For a=572, m=16381 :  3.1413698833786614
