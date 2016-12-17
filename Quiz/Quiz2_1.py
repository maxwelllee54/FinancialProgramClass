# Author: Yanzhe Li
# Date: 09/16/2016
# Description: Use Hart’s algorithm to compute cumulative standard normal distribution

from math import fabs, exp
from scipy import stats

# To compare the accuracy between the self written function calculateStandardNormalCDF() and stats.norm.cdf() from the scipy package
def main():
    test = [-0.02, 0.02, -0.8, 0.182, 8.0]
    for x in test:
        print('The test number is {:+.4f}'.format(x))
        print('The self written function gives the result of {:.16f}'.format(calculateStandardNormalCDF(x)))
        print('The stats.norm.cdf() gives the result of {:.16f}'.format(stats.norm.cdf(x)))
        print('The difference between two functions is {:.16f}\n'.format(fabs(calculateStandardNormalCDF(x) - stats.norm.cdf(x))))

# Use Hart's algorithm to compute the cumulative standard normal distribution
def calculateStandardNormalCDF(x):
    t = fabs(x)
    t1 = 7.07106781186547
    t2 = 37
    t3 = 2.506628274631

    a = [0.0352624965998911, 0.700383064443688, 6.37396220353165, 33.912866078383, \
         112.079291497871, 221.213596169931, 220.206867912376]
    b = [0.0883883476483184, 1.75566716318264, 16.064177579207, 86.7807322029461, \
         296.564248779674, 637.333633378831, 793.826512519948, 440.413735824752]

    A = ((((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5])* t + a[6])
    B = (((((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + b[5]) * t + b[6]) * t + b[7])
    C = t + 1/(t + 2/(t + 3/(t + 4/(t + 0.65))))

    if x <= 0:
        if t < t1:
            return  exp(-t**2/2) * A / B
        elif t <= t2:
            return  exp(-t**2/2) / (t3 * C)
        else:
            return 0
    else:
        return 1 - calculateStandardNormalCDF(-x)


main()