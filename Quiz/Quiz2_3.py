# Author: Yanzhe Li
# Date: 09/16/2016
# Description: Polish Black Scholes Model

from math import log, sqrt, exp
from scipy import stats


# main function to test bs_pricing and bsm_pricing function
def main():
    print('#' * 20, 'FIRST TEST', '#' * 20)
    print('\nAn european call: S = 89, K = 100, T = 0.5, r = 0.02, σ = 0.2\n')
    print('The price for this call option is ${:.4f}\n'.format(bs_pricing(89, 100, 0.5, 0.02, 0.2, 'call')))

    print('#' * 20, 'SECOND TEST', '#' * 20)
    print('\nAn european put: S = 102.5, K = 88.5, T = 0.25, r = 0.03, σ = 0.3\n')
    print('The price for this put option is ${:.4f}\n'.format(bs_pricing(102.5, 88.5, 0.25, 0.03, 0.3, 'put')))

    print('#' * 20, 'THIRD TEST', '#' * 20)
    print('\nAn european call with continuous dividend yield: S = 50, K = 80, T = 5/12, r = 0.1, σ = 0.35, q = 0.05\n')
    print('The price for this call option is ${:.4f}\n'.format(bsm_pricing(50, 80, 5 / 12, 0.1, 0.05, 0.35, 'call')))

    print('#' * 20, 'FOURTH TEST', '#' * 20)
    print('\nAn european put on stock index with a cost-of-carry: S = 80, K = 75, T = 5/12, r = 0.1, σ = 0.2, q = 0.07\n')
    print('The price for this put option is ${:.4f}\n'.format(bsm_pricing(80, 75, 5 / 12, 0.1, 0.07, 0.20, 'put')))


# a function to handle both call and put option for BS model
def bs_pricing(S, K, T, r, sigma, option_type):

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type in ['Call', 'call', 'CALL']:

        return S * stats.norm.cdf(d1) - K* exp(-r * T)* stats.norm.cdf(d2)

    elif option_type in ['Put', 'put', 'PUT']:

        return K * exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

    else:
        raise TypeError('the option_type argument must be either "call" or "put"')


# a function to handle both call and put option for BSM model
def bsm_pricing(S, K, T, r, q, sigma, option_type):

    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type in ['Call', 'call', 'CALL']:

        return S * exp(-q * T) * stats.norm.cdf(d1) - K * exp(-r * T) * stats.norm.cdf(d2)

    elif option_type in ['Put', 'put', 'PUT']:

        return K * exp(-r * T) * stats.norm.cdf(-d2) - S * exp(-q * T) * stats.norm.cdf(-d1)

    else:
        raise TypeError('the option_type argument must be either "call" or "put"')


main()