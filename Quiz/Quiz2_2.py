# Author: Yanzhe Li
# Date: 09/16/2016
# Description: Polish Monte Carlo simulations

from numpy import *
import pylab as pl
import time


# to test the self written function with two examples and plot the stimulation result of the first certain trials
def main():
    time_start = time.clock()
    test_result_call = mc_pricing(50, 80, 5 / 12, 0.1, 0.35, 'call', 10 ** 7)
    test_result_put = mc_pricing(80, 75, 5 / 12, 0.1, 0.2, 'put', 10 ** 7)

    print('1. The price of this european call is ${:.2f}.'.format(test_result_call[0]))
    print('2. The price of this european put is ${:.2f}.'.format(test_result_put[0]))

    print('The program takes {:.4f} seconds to run\n'.format(time.clock() - time_start))

    pl.plot(test_result_call[1])
    pl.plot(test_result_put[1])
    pl.legend(('call option', 'put option'))
    pl.show()


# pricing function to calculate both call and put option price by Monte Carlo simulations
def mc_pricing(S, K, T, r, sigma, option_type, no_trial):
    """
    :param S:           initiate stock price
    :param K:           strike price
    :param T:           time to maturity
    :param r:           riskless interest rate
    :param sigma:       risk volatility
    :param option_type: call or put
    :param no_trial:    number of stimulations
    :return:            option call or put price
                        all possible stock prices of the stimulation
    """

    # random.seed(999)

    z = random.standard_normal(no_trial)

    ST = S * exp((r - 0.5 * sigma ** 2) * T + sigma * sqrt(T) * z)

    if option_type in ['Call', 'call', 'CALL']:

        payoff = maximum(ST - K, 0)

    elif option_type in ['Put', 'put', 'PUT']:

        payoff = maximum(K - ST, 0)

    else:
        raise TypeError('the option_type argument must be either "call" or "put"')

    price = exp(-r * T) * sum(payoff) / no_trial

    price_plot = []

    step = 10
    count = 1

    while count < 500000:
        price_plot.append(exp(-r * T) * sum(payoff[:count]) / count)
        count += step

    return price, price_plot


main()
