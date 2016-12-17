# Author: Yanzhe Li
# Date: 09/16/2016
# Description: Polish Monte Carlo simulations

from numpy import *
import pylab as pl
import time
import threading

# use the threading class to calculate both call and put option price by Monte Carlo simulations
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
    z = random.standard_normal(no_trial)

    ST = S * exp((r - 0.5 * sigma ** 2) * T + sigma * sqrt(T) * z)

    if option_type in ['Call', 'call', 'CALL']:

        payoff = maximum(ST - K, 0)

    elif option_type in ['Put', 'put', 'PUT']:

        payoff = maximum(K - ST, 0)

    else:
        raise TypeError('the option_type argument must be either "call" or "put"')

    price = exp(-r * T) * sum(payoff) / no_trial

    return price, ST


# to plot the first n points of the list
def my_plot(lst, n=0):

    x = [x for x in range(n)]
    y = lst[:n]
    pl.plot(x, y)

time_start = time.clock()
no_trials = 100

for i in range(no_trials):
    test_result_call = threading.Thread(target = mc_pricing, args = (50, 80, 5 / 12, 0.1, 0.35, 'call', i))
    test_result_put = threading.Thread(target = mc_pricing, args = (80, 75, 5 / 12, 0.1, 0.2, 'put', i))
    test_result_call.run()
    test_result_put.run()


#print('1. The price of this european call is ${:.2f}.'.format(test_result_call[0]))
#print('2. The price of this european put is ${:.2f}.'.format(test_result_put[0]))
print('The running time is {}'.format(time.clock() - time_start))

#no_points = 5
#my_plot(test_result_call[1], no_points)
#my_plot(test_result_put[1], no_points)
#pl.legend(('call option', 'put option'))
#pl.show()
