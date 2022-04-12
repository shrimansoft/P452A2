# Assuming unit radius for the two cylinders
from library import mlcg

# Function to calculate the common enclosed volume
# NOTE: the function is even about x=0
def steinmetz(x):
    return 4 * (1 - x**2)


def monteCarlo(func, N, lims):
    # Generate list of N random points between lims
    xrand = mlcg(234.34, 65, 1, N)

    summation = 0
    for i in range(N):
        summation += func(xrand[i])

    total = 1 / float(N) * summation

    return total


volume = monteCarlo(steinmetz, 10000, (0, 1))
print("The enclosed volume between the cylinders is {} units.".format(2 * volume))

# ---------------------------------------* OUTPUT *------------------------------------
# The enclosed volume between the cylinders is 5.322956716258391 units.
