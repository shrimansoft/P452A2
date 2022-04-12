import numpy as np
import math
import random

import matplotlib.pyplot as plt


# Function to generate random numbers
# multiplicative linear congruential generator
def mlcg(Xo, m, a, c, randomNums, N):
    """_summary_

    Args:
        Xo (_type_): _description_
        m (_type_): _description_
        a (_type_): _description_
        c (_type_): _description_
        randomNums (_type_): _description_
        N (_type_): _description_

    Returns:
        _type_: _description_
    """
    randomNums[0] = Xo
    # numbers of random numbers
    for i in range(1, N):
        randomNums[i] = ((randomNums[i - 1] * a+c) % m)
    return randomNums


# DFT of a 1D real-valued signal x
def DFT(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(e, x)

    return X

# Inverse DFT of a 1D real-valued signal x



def InDFT(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)

    iX = np.dot(e, x)/N

    return iX
# chi2 fit


def chi2(x, y, sgm):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        sgm (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = len(x)
    chi2 = 0
    s = 0
    sx = 0
    sy = 0
    sxy = 0
    sxx = 0
    syy = 0
    a = 0
    b = 0
    for i in range(n):
        s += 1/(sgm[i]**2)
        sx += x[i]/(sgm[i]**2)
        sy += y[i]/(sgm[i]**2)
        sxx += x[i]**2/sgm[i]**2
        sxy += x[i]*y[i]/sgm[i]**2
        syy += y[i]*y[i]/sgm[i]**2
    Delta = s*sxx-(sx)**2
    a = (sxx*sy-sx*sxy)/Delta
    b = (s*sxy-sx*sy)/Delta
    sgma2 = sxx/Delta
    sgmb2 = s/Delta
    covab = -sx/Delta
    r2 = sxy/(sxx*syy)
    for i in range(n):
        chi2 += ((y[i]-a-b*x[i])/sgm[i])**2
    return chi2, sgma2, sgmb2, covab, r2, a, b


# jack knife


def jackknife(x, func):
    """Jackknife estimate of the estimator func"""
    n = len(x)
    idx = np.arange(n)
    return np.sum(func(x[idx != i]) for i in range(n))/float(n)


def jackknife_var(x, func):
    """Jackknife estiamte of the variance of the estimator func."""
    n = len(x)
    idx = np.arange(n)
    j_est = jackknife(x, func)
    return (n-1)/(n + 0.0) * np.sum((func(x[idx != i]) - j_est)**2.0
                                    for i in range(n))


def normalize(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n


def powerr(a):
    """_summary_

    Args:
        a (_type_): _description_

    Returns:
        _type_: _description_
    """
    x = np.array([1, 1, 1, 1, 1])

    for i in range(8):
        x = np.dot(a, x)
        lambda_1, x = normalize(x)

    return lambda_1, x

# Power Iteration................Eigenvalue vector problem



def __find_p(x):
    """_summary_


    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.argwhere(np.isclose(np.abs(x), np.linalg.norm(x, np.inf))).min()


def __iterate(A, x, p):
    """_summary_

    Args:
        A (_type_): _description_
        x (_type_): _description_
        p (_type_): _description_

    Returns:
        _type_: _description_
    """

    y = np.dot(A, x)
    μ = y[p]
    p = __find_p(y)
    error = np.linalg.norm(x - y / y[p],  np.inf)
    x = y / y[p]

    return (error, p, μ, x)


def power_method(A, tolerance=1e-10, max_iterations=10000):
    """_summary_

    Args:
        A (_type_): _description_
        tolerance (_type_, optional): _description_. Defaults to 1e-10.
        max_iterations (int, optional): _description_. Defaults to 10000.

    Returns:
        _type_: _description_
    """

    n = A.shape[0]
    x = np.ones(n)

    p = __find_p(x)

    error = 1

    x = x / x[p]

    for _ in range(max_iterations):

        if error < tolerance:
            break

        error, p, μ, x = __iterate(A, x, p)

    return (μ, x)


# Jacobi method ............Eigenvalue vector problem
# module jacobi
''' lam,x = jacobi(a,tol = 1.0e-9).
    Solution of std. eigenvalue problem [a]{x} = lam{x}
    by Jacobi's method. Returns eigenvalues in vector {lam}
    and the eigenvectors as columns of matrix [x].
'''


def jacobi(a, tol=1.0e-9):  # Jacobi method
    """_summary_

    Args:
        a (_type_): _description_
        tol (_type_, optional): _description_. Defaults to 1.0e-9.
    """



    def maxElem(a):  # Find largest off-diag. element a[k,l]
        """_summary_

        Args:
            a (_type_): _description_

        Returns:
            _type_: _description_
        """
        n = len(a)
        aMax = 0.0
        for i in range(n-1):
            for j in range(i+1, n):
                if abs(a[i, j]) >= aMax:
                    aMax = abs(a[i, j])
                    k = i
                    l = j
        return aMax, k, l

    def rotate(a, p, k, l):  # Rotate to make a[k,l] = 0
        n = len(a)
        aDiff = a[l, l] - a[k, k]
        if abs(a[k, l]) < abs(aDiff)*1.0e-36:
            t = a[k, l]/aDiff
        else:
            phi = aDiff/(2.0*a[k, l])
            t = 1.0/(np.abs(phi) + math.sqrt(phi**2 + 1.0))
            if phi < 0.0:
                t = -t
        c = 1.0/math.sqrt(t**2 + 1.0)
        s = t*c
        tau = s/(1.0 + c)
        temp = a[k, l]
        a[k, l] = 0.0
        a[k, k] = a[k, k] - t*temp
        a[l, l] = a[l, l] + t*temp
        for i in range(k):      # Case of i < k
            temp = a[i, k]
            a[i, k] = temp - s*(a[i, l] + tau*temp)
            a[i, l] = a[i, l] + s*(temp - tau*a[i, l])
        for i in range(k+1, l):  # Case of k < i < l
            temp = a[k, i]
            a[k, i] = temp - s*(a[i, l] + tau*a[k, i])
            a[i, l] = a[i, l] + s*(temp - tau*a[i, l])
        for i in range(l+1, n):  # Case of i > l
            temp = a[k, i]
            a[k, i] = temp - s*(a[l, i] + tau*temp)
            a[l, i] = a[l, i] + s*(temp - tau*a[l, i])
        for i in range(n):      # Update transformation matrix
            temp = p[i, k]
            p[i, k] = temp - s*(p[i, l] + tau*p[i, k])
            p[i, l] = p[i, l] + s*(temp - tau*p[i, l])

    n = len(a)
    maxRot = 5*(n**2)       # Set limit on number of rotations
    p = np.identity(n)*1.0     # Initialize transformation matrix
    for i in range(maxRot):  # Jacobi rotation loop
        aMax, k, l = maxElem(a)
        if aMax < tol:
            return np.diagonal(a), p
        rotate(a, p, k, l)
    print('Jacobi method did not converge')


# Conjugate Gradient
def conjGrad(A, b, x, tol):
    """_summary_

    Args:
        A (_type_): _description_
        b (_type_): _description_
        x (_type_): _description_
        tol (_type_): _description_

    Returns:
        _type_: _description_
    """
    N = 100000
    r = b - A.dot(x)
    p = r.copy()
    er = np.array([])
    kk = np.array([])
    for i in range(N):
        Ap = A.dot(p)
        alpha = np.dot(p, r)/np.dot(p, Ap)
        x = x + alpha*p
        r = b - A.dot(x)
        er = np.append(er, np.sum((r**2)))
        kk = np.append(kk, i)
        if np.sqrt(np.sum((r**2))) < tol:
            # print('Itr:', i)
            break
        else:
            beta = -np.dot(r, Ap)/np.dot(p, Ap)
            p = r + beta*p
    return x, er, kk


# Gauss-Seidel iterative method


def gauss_seidel(A, b, tolerance, max_iterations=10000):
    """_summary_

    Args:
        A (_type_): _description_
        b (_type_): _description_
        tolerance (_type_): _description_
        max_iterations (int, optional): _description_. Defaults to 10000.

    Returns:
        _type_: _description_
    """

    x = np.zeros_like(b, dtype=np.double)
    er = np.array([])
    kk = np.array([])
    # Iterate
    for k in range(max_iterations):
        x_old = x.copy()

        # Loop over rows
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, (i+1):], x_old[(i+1):])) / A[i, i]
        er = np.append(er, np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf))
        kk = np.append(kk, k)
        # Stop condition
        if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerance:
            break

    return x, er, kk

# LU decomposion function break a given matrix into upper and lower triangle matrix


def ludecom(a):
    """_summary_

    Args:
        a (_type_): _description_

    Returns:
        _type_: _description_
    """
    fff = len(a[1])
    # Inisialize upper and lower matrix
    with open('l_and_u.txt') as f:
        fff = len(a[1])
        u = []
        for i in range(0, fff):
            u.append(list(map(float, f.readline().split())))
    with open('l_and_u.txt') as f:
        l = []
        for i in range(0, fff):
            l.append(list(map(float, f.readline().split())))
    n = len(u)

    # lets start transformation of upper and lower triangle matrix
    for i in range(0, n):
        # upper triangle matrix
        # u[i][j]=a[i][j]-sum(from k=0 to i)(l[i][k]*u[k][j])
        for j in range(0, n):
            # sum(from j=0 to i)(l[i][k]*u[k][j])
            sum = 0
            for k in range(0, i):
                sum += l[i][k]*u[k][j]
            u[i][j] = a[i][j]-sum

        # lower triangle matrix
        # l[i][j] = (1/u[j][j])*(a[i][j]-sum(from k=0 to i)(l[i][k]*u[k][j]))
        # to get the lower triangle matrix just swap row indics to column indics
        # For example l[i][j] to l[j][i] and same for a[i][j]
        # this is because we have to solve column-wise for lower triangle
        for j in range(0, n):
            if i == j:
                l[i][i] = 1
            else:
                sum = 0
                for k in range(0, i):
                    sum += l[j][k]*u[k][i]
                l[j][i] = (a[j][i]-sum)/u[i][i]
    return l, u


def jacobimat(A, b, itr):
    """_summary_

    Args:
        A (_type_): _description_
        b (_type_): _description_
        itr (_type_): _description_

    Returns:
        _type_: _description_
    """
    ITERATION_LIMIT = itr
    # prints the system
    '''
    print("System:")
    for i in range(A.shape[0]):
        row = ["{}*x{}".format(A[i, j], j + 1) for j in range(A.shape[1])]
        print(f'{" + ".join(row)} = {b[i]}')
    print()
    '''
    er = np.array([])
    kk = np.array([])
    x = np.zeros_like(b)
    for it_count in range(ITERATION_LIMIT):
        # if it_count != 0:
        # print("Iteration {0}: {1}".format(it_count, x))
        x_new = np.zeros_like(x)

        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
            if x_new[i] == x_new[i-1]:
                break
        er = np.append(er, x_new-x)
        kk = np.append(kk, it_count)
        if np.allclose(x, x_new, atol=1e-4, rtol=0.):
            break

        x = x_new
    error = np.dot(A, x) - b
    return x, er, kk


'''
#####################################################################
#################################################################
####################################################################
##################################################################
'''
'''
#####################################################################
#################################################################
####################################################################
##################################################################
'''
# Montecarlo integration


def montecarlo(a, b, n, f):
    """_summary_

    Args:
        a (_type_): _description_
        b (_type_): _description_
        n (_type_): _description_
        f (_type_): _description_

    Returns:
        _type_: _description_
    """
    Fn = 0
    sigmaf = 0
    for i in range(1, n+1):
        # random number
        x = a+(b-a)*random.random()
        # intergral
        Fn += ((b-a)/n)*f(x)
        # error
        sigmaf += math.sqrt(((1/n)*(f(x)**2))-(((1/n)*f(x))**2))
    return Fn, sigmaf


# a,b limit of integration
def montecarlo_ran_nums(randnums, n, f, a, b):
    """_summary_

    Args:
        randnums (_type_): _description_
        n (_type_): _description_
        f (_type_): _description_
        a (_type_): _description_
        b (_type_): _description_

    Returns:
        _type_: _description_
    """
    Fn = 0
    sigmaf = 0
    for i in range(1, n+1):
        # random number
        x = a+(b-a)*randnums[i-1]
        # intergral
        Fn += ((b-a)/n)*f(x)
        # error
        sigmaf += math.sqrt(((1/n)*(f(x)**2))-(((1/n)*f(x))**2))
    return Fn, sigmaf


'''################################################################################################'''
'''##################################################################################################'''

'''################################################################################################'''
'''##################################################################################################'''

'''################################################################################################'''
'''##################################################################################################'''

'''################################################################################################'''
'''##################################################################################################'''

'''################################################################################################'''
'''##################################################################################################'''


# polynomial fitting
def data(file, char_):
    """
    Open a file that contains
    data to fit | Format (x,y)
    char_ : separator character
           ' '     <== Space delimited
           '\t'    <== Tabs delimited
           ','     <== Comma delimited
    """
    data = open(file, 'r')
    line = [line.rstrip().split(char_)
            for line in data]
    x = column(line, 0)
    y = column(line, 1)
    for i in range(len(x)):
        x[i] = float(x[i])
        y[i] = float(y[i])
    return x, y


def column(a, i):
    """
    Returns a specific column
    of a multidimensional list
    """
    return [row[i] for row in a]


def transpose(a):
    """
    Returns the transpose of a
    mutidimensional list
    """
    trans_a = []
    for i in range(len(a[0])):
        trans_a.append(column(a, i))
    return trans_a


def polynomial(M, i):
    """
    Helps to create the matrix A
    """
    row = [1]
    for exp in range(1, M+1):
        row.append(i ** exp)
    return row


def matrix_A(x, M):
    """
    Create matrix A to fit data
    to a polynomial of order M
    """
    N = len(x)
    matrix_A = []
    for i in x:
        matrix_A.append(polynomial(M, i))
    return matrix_A


def plots(x, y1, y2):
    """
    Function to plot
    """
    fig = plt.figure(figsize=(4, 4))
    a = fig.add_subplot(1, 1, 1)
    a.plot(x, y1, 'k|', marker='o', markersize=4, markeredgewidth=1,
           markeredgecolor='k', markerfacecolor='w', label='Original data')
    a.plot(x, y2, 'r-', linewidth=1, label='Fitted')
    a.set_ylabel('Y')
    a.set_xlabel('X')
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.legend(ncol=2, loc='upper left',
             fontsize=9, frameon=False)
    fig.tight_layout()
    return


def function(x, pol_coeff):
    """
    Function to evaluate the polynomial from fitting process
    """
    res = 0
    for exp, coeff in enumerate(pol_coeff):
        res = res + coeff[0] * x ** exp
    return res


def fitting(file, char_, M, plot=True):
    """
    Main function:
    File   : file name containing data
    char_  : character separator
    M      : order of polynomial to fit data
    plot   : If True ==> Plot , if False ==> pass
    """
    points = data(file, char_)
    x, Y = points[0], points[1]
    y = [Y[i: (i+1)] for i in range(len(Y))]
    A_transpose = transpose(matrix_A(x, M))
    Matrix_S = matrix_mult(A_transpose, matrix_A(x, M))
    vector_Z = matrix_mult(A_transpose, y)
    S_inverse = inverse(Matrix_S)
    pol_coeff = matrix_mult(S_inverse, vector_Z)
    y_calc = []
    for value in x:
        y_calc.append(function(value, pol_coeff))
    if plot is True:
        plots(x, Y, y_calc)
        plt.show()
    elif plot is False:
        pass
    y_mean = sum(Y) / len(Y)
    sum_upper = 0
    sum_lower = 0
    for i in range(len(x)):
        sum_upper = sum_upper + (y_calc[i] - y_mean) ** 2
        sum_lower = sum_lower + (Y[i] - y_mean) ** 2
    R_2 = sum_upper / sum_lower
    return {'Coefficients': pol_coeff, "Pearson's r": math.sqrt(R_2)}


def expofitting(file, char_):

    (x, y) = data(file, char_)
    # q=xy, w=xylny, e=y, r=x2y, t=xy, u=ylny
    q = 0
    w = 0
    e = 0
    r = 0
    t = 0
    u = 0
    p = 0
    for i in range(len(x)):
        q += x[i]*y[i]
        w += x[i]*y[i]*math.log(y[i])
        e += y[i]
        r += x[i]**2*y[i]
        t += x[i]*y[i]
        u += y[i]*math.log(y[i])
        p += x[i]
    a = (r*u-q*w)/(e*r-q**2)
    b = (e*w-q*u)/(e*r-q**2)
    ssx = 0
    ssy = 0
    for i in range(len(x)):
        ssx += (x[i]-p/len(x))**2
        ssy += (y[i]-e/len(x))**2
    sxy = q-len(x)*p/len(x)*e/len(x)
    r = sxy**2/(ssx*ssy)

    return {'Coefficients': [a, b], "Pearson's r": math.sqrt(r)}


'''################################################################################################'''
'''##################################################################################################'''

'''################################################################################################'''
'''##################################################################################################'''

'''################################################################################################'''
'''##################################################################################################'''

'''################################################################################################'''
'''##################################################################################################'''

'''################################################################################################'''
'''##################################################################################################'''
# solutions of matrix without augmentation


def partialpivot(a, b):
    """_summary_

    Args:
        a (_type_): _description_
        b (_type_): _description_

    Returns:
        _type_: _description_
    """
    r = len(b)
    for k in range(0, r-1):
        if a[k][k] == 0:
            for i in range(k+1, r):
                if abs(a[i][k]) > abs(a[k][k]):
                    store = a[k]
                    a[k] = a[i]
                    a[i] = store
                    store = b[k]
                    b[k] = b[i]
                    b[i] = store
    return a, b


def gaussjordan(a, b):

    partialpivot(a, b)
    r = len(b)
    for k in range(0, r):
        pivot = a[k][k]
        # for the pivot row divide by akk
        for j in range(k, r):
            a[k][j] = a[k][j]/pivot
        b[k] = b[k]/pivot
        # for the non pivot row
        for i in range(0, r):
            if i == k or a[i][k] == 0:
                continue
            factor = a[i][k]
            # substraction
            for d in range(k, r):
                a[i][d] = a[i][d]-factor * a[k][d]
            b[i] = b[i]-factor * b[k]
    return a, b


def forback(a, b):
    l = ludecom(a)[0]
    u = ludecom(a)[1]
    n = len(u)
    # AX=B
    # A=LU => LUX=B
    # let UX=Z => LZ=B
    # Solve the system LZ=B by forward substitution
    # create zero matrix z
    z = []
    for i in range(0, n):
        z.append(0)
    # z1=b1/l11 and zi=(bi-sum(j=1 to i-1)lij*zj)/lii
    for i in range(0, n):
        if i == 0:
            z[i] = b[i]/l[i][i]
        else:
            sum = 0
            for j in range(0, i):
                sum += l[i][j]*z[j]
            z[i] = (b[i]-sum)/l[i][i]

    # Solve the system UX = Z by backward substitution
    # create zero matrix x
    x = []
    for i in range(0, n):
        x.append(0)
    # xn=zn and xi=(zi-sum(uij*uj))/uii
    i = n-2
    x[n-1] = z[n-1]/u[n-1][n-1]
    while i >= 0:
        sum = 0
        for j in range(i, n):
            sum += u[i][j]*x[j]
        x[i] = (z[i]-sum)/u[i][i]
        i = i-1

    return z, x


# from last assignment
def partialpivot2(a):
    r = len(a)
    for k in range(0, r-1):
        if a[k][k] == 0:
            for i in range(k+1, r):
                if abs(a[i][k]) > abs(a[k][k]):
                    store = a[k]
                    a[k] = a[i]
                    a[i] = store
    return a


def augment(a, b):
    for i in range(0, len(a)):
        try:
            for j in range(0, len(b[0])):
                a[i].append(b[i][j])
        except TypeError:
            a[i].append(b[i])
    return a


def gaussjordan2(a, b):
    augment(a, b)
    partialpivot2(a)
    row = len(a)
    col = len(a[0])
    for k in range(0, row):
        pivot = a[k][k]
        # for the pivot row divide by akk
        for j in range(k, col):
            a[k][j] = a[k][j]/pivot
        # for the non pivot row
        for i in range(0, row):
            if i == k or a[i][k] == 0:
                continue
            factor = a[i][k]
            # substraction
            for d in range(k, col):
                a[i][d] = a[i][d]-factor * a[k][d]
    return a


# b is identity matrix
def inversegj(a, b):
    gaussjordan2(a, b)
    # have to change the solution according to need
    sol = [[], [], [], [], [], []]
    for i in range(0, len(a)):
        for j in range(len(a), len(a[1])):
            sol[i].append(a[i][j])
    return sol

# check and does partialpivot by itself and return ready to use matrix


def makeready(a):
    # try to run ludecom function and it find ZeroDivisionError then pivoting
    try:
        ludecom(a)
    except ZeroDivisionError:
        a = partialpivot2(a)
    return a


# find the determinant
def determinant(a):
    u = ludecom(a)[1]
    det = 1
    for i in range(0, len(u)):
        det = det*u[i][i]
    return det


# Inverse by LU decomposion
# AA^-1=I
# A*(first column)=first column............solve for all columns
def inverselu(a, b):
    # here b is identity matrix
    # Inisialize ainverse matrix
    fff = len(a[1])
    with open('l_and_u.txt') as f:
        ain = []
        for i in range(0, fff):
            ain.append(list(map(float, f.readline().split())))

    for i in range(0, len(ain)):
        c = []
        for j in range(0, len(ain)):
            # send i th column to solve
            c.append(b[j][i])
        x = forback(a, c)[1]
        # put solution of ith column into ainv
        for j in range(0, len(ain)):
            ain[j][i] = x[j]
    return ain


'''
#####################################################################
#################################################################
####################################################################
##################################################################
'''


def matrix_mult(a, b):
    """
    Matrix multiplication
    """
    zip_b = zip(*b)
    zip_b = list(zip_b)
    matrix = [[sum(a_i * b_i for a_i,
                   b_i in zip(row_a, col_b))
               for col_b in zip_b] for row_a in a]
    return matrix


def Identity(n):
    """
    Create a identity matrix
    """
    result = d_2_List(n, n)
    for i in range(n):
        result[i][i] = 1
    return result


def d_2_List(rows, cols):
    """
    Auxiliar function
    to calculate inverse
    """
    a = []
    for row in range(rows):
        a += [[0]*cols]
    return a


def S_Matrix(m, row, k):
    """
    Square matrix
    """
    n = len(m)
    r_Oper = Identity(n)
    r_Oper[row][row] = k
    return matrix_mult(r_Oper, m)


def add_S_Matrix(m, s_row, k, t_row):
    """
    Add rows of
    square matrix
    """
    n = len(m)
    r_oper = Identity(n)
    r_oper[t_row][s_row] = k
    return matrix_mult(r_oper, m)


def inverse(m):
    """
    Inverse of matrix m
    """
    n = len(m)
    assert(len(m) == len(m[0]))
    inverse = Identity(n)
    for col in range(n):
        d_row = col
        assert(m[d_row][col] != 0)
        k = 1 / m[d_row][col]
        m = S_Matrix(m, d_row, k)
        inverse = S_Matrix(inverse, d_row, k)
        s_row = d_row
        """
        Gauss Jordan Elimination
        """
        for t_row in range(n):
            if (s_row != t_row):
                k = -m[t_row][col]
                m = add_S_Matrix(m, s_row,
                                 k, t_row)
                inverse = add_S_Matrix(inverse,
                                       s_row, k,
                                       t_row)
    return inverse


'''################################################################################################'''
'''##################################################################################################'''

'''################################################################################################'''
'''##################################################################################################'''

'''################################################################################################'''
'''##################################################################################################'''

# fitting by modified chebyshev fucntion


def polynomial_mod(M, i):
    row = [1]
    for exp in range(1, M+1):
        if exp == 0:
            row.append(1)
        elif exp == 1:
            row.append(2*i-1)
        elif exp == 2:
            row.append(8*i**2-8*i+1)
        elif exp == 3:
            row.append(32*i**3-48*i**2+18*i-1)
    return row


def matrix_A_mod(x, M):
    """
    Create matrix A to fit data
    to a polynomial of order M
    """
    N = len(x)
    matrix_A = []
    for i in x:
        matrix_A.append(polynomial_mod(M, i))
    return matrix_A


def function_mod(x, pol_coeff):
    """
    Function to evaluate the polynomial from fitting process
    """
    res = 0
    for exp, coeff in enumerate(pol_coeff):
        if exp == 0:
            res = res + coeff[0] * (1)
        elif exp == 1:
            res = res + coeff[0] * (2*x-1)
        elif exp == 2:
            res = res + coeff[0] * (8*x**2-8*x+1)
        elif exp == 3:
            res = res + coeff[0] * (32*x**3-48*x**2+18*x-1)
    return res


def fitting_modified(file, char_, M, plot=True):
    """
    Main function:
    File   : file name containing data
    char_  : character separator
    M      : order of polynomial to fit data
    plot   : If True ==> Plot , if False ==> pass
    """
    points = data(file, char_)
    x, Y = points[0], points[1]
    y = [Y[i: (i+1)] for i in range(len(Y))]
    A_transpose = transpose(matrix_A_mod(x, M))
    Matrix_S = matrix_mult(A_transpose, matrix_A_mod(x, M))
    vector_Z = matrix_mult(A_transpose, y)
    S_inverse = inverse(Matrix_S)
    pol_coeff = matrix_mult(S_inverse, vector_Z)
    y_calc = []
    for value in x:
        y_calc.append(function_mod(value, pol_coeff))
    if plot is True:
        plots(x, Y, y_calc)
        plt.show()
    elif plot is False:
        pass
    y_mean = sum(Y) / len(Y)
    sum_upper = 0
    sum_lower = 0
    for i in range(len(x)):
        sum_upper = sum_upper + (y_calc[i] - y_mean) ** 2
        sum_lower = sum_lower + (Y[i] - y_mean) ** 2
    R_2 = sum_upper / sum_lower
    return {'Coefficients': pol_coeff, "Pearson's r": math.sqrt(R_2)}
