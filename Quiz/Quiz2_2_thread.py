# Author: Yanzhe Li
# Date: 09/16/2016
# Description: Polish Monte Carlo simulations

from numpy import *
import pylab as pl
import time
import threading


# Thread class to calculate both call and put option price by Monte Carlo simulations
class mc_thread(threading.Thread):
    def __init__(self, S, K, T, r, sigma, option_type, no_trial):
        threading.Thread.__init__(self)
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.no_trial = no_trial

    def run(self):
        ST = self.S * exp((self.r - 0.5 * self.sigma ** 2) * self.T + self.sigma * sqrt(self.T) * random.standard_normal(self.no_trial))

        if self.option_type in ['Call', 'call', 'CALL']:

            payoff = maximum(ST - self.K, 0)

        elif self.option_type in ['Put', 'put', 'PUT']:

            payoff = maximum(self.K - ST, 0)

        else:
            raise TypeError('the option_type argument must be either "call" or "put"')

        price = exp(-self.r * self.T) * sum(payoff) / self.no_trial

        return price, ST

# to plot the first n points of the list
def my_plot(lst, n=0):
    x = [x for x in range(n)]
    y = lst[:n]
    pl.plot(x, y)

time_start = time.clock()
no_trial = 1000

for i in range(no_trial):
    call_option = mc_thread(50, 80, 5 / 12, 0.1, 0.35, 'call', )
    put_option = mc_thread(80, 75, 5 / 12, 0.1, 0.2, 'put', )

call_result = call_option.run()
put_result = put_option.run()

print('1. The price of this european call is ${:.2f}.'.format(call_result[0]))
print('2. The price of this european put is ${:.2f}.'.format(put_result[0]))

print('The program takes {:.4f} seconds to run\n'.format(time.clock() - time_start))

