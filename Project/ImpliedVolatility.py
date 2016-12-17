import math as e
from scipy import stats
from datetime import date
import time


class ImpliedVolatility():
    def __init__(self, S, K, T, r, sigma, cStar, optionType, err = 1e-15):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.cStar = cStar
        self.optionType = optionType
        self.err = float(err)

    def bsmValue(self, sigma = None):
        if sigma == None:
            sigma = self.sigma

        d1 = (e.log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * e.sqrt(self.T))
        d2 = d1 - sigma * e.sqrt(self.T)

        if self.optionType in ['Call', 'call', 'CALL']:

            return self.S * stats.norm.cdf(d1) - self.K * e.exp(-self.r * self.T) * stats.norm.cdf(d2)

        elif self.optionType in ['Put', 'put', 'PUT']:

            return self.K * e.exp(-self.r * self.T) * stats.norm.cdf(-d2) - self.S * stats.norm.cdf(-d1)

        else:
            raise TypeError('the option_type argument must be either "call" or "put"')

    def bsmVega(self):
        d1 = (e.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * e.sqrt(self.T))
        vega = self.S * stats.norm.pdf(d1) * e.sqrt(self.T)
        return vega

    def bsmVomma(self):
        d1 = (e.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * e.sqrt(self.T))
        d2 = d1 - self.sigma * e.sqrt(self.T)

        return self.bsmVega() * d1 * d2 / self.sigma

    def f(self, sigma = None):
        return self.bsmValue(sigma) - self.cStar

    def bsmMuller(self, x0, x1, x2):

        f0, f1, f2 = self.f(x0), self.f(x1), self.f(x2)
        h0, h1 = x0 - x2, x1 - x2
        e0, e1 = f0 - f2, f1 - f2
        det = h0 * h1 * (h0 - h1)
        A = (e0 * h1 - h0 * e1) / det
        B = (h0 ** 2 * e1 - h1 ** 2 * e0) / det
        C = f2

        if B < 0:
            x3 = x2 - 2 * C / (B - e.sqrt(B ** 2 - 4 * A * C))
        else:
            x3 = x2 - 2 * C / (B + e.sqrt(B ** 2 - 4 * A * C))

        return x3

    def bsmBisectionVol(self):
        lower = self.err
        upper = 1
        middle = (lower + upper)/2
        while e.fabs(self.f(middle)) > self.err:
            if self.f(lower) * self.f(middle) < 0:
                upper = middle
                middle = (lower + upper) / 2
            else:
                lower = middle
                middle = (lower + upper) / 2
        return middle

    def bsmNewtonVol(self):
        while self.f() > self.err:
            self.sigma = self.sigma - self.f()/self.bsmVega()
        return self.sigma

    def bsmMullerBisectionVol(self):
        lower = self.err
        upper = 1
        middle = (lower + upper) / 2

        while e.fabs(self.f(middle)) > self.err:
            muller = self.bsmMuller(lower, upper, middle)
            if self.f(lower) * self.f(middle) < 0:
                upper = middle
            else:
                lower = middle

            if muller < lower or muller > upper:
                middle = (lower + upper) / 2
            else:
                middle = muller

        return middle

    def bsmHalley(self):
        while self.f() > self.err:
            self.sigma = self.sigma + (-self.bsmVega() + e.sqrt(self.bsmVega() ** 2 - 2 * self.f() * self.bsmVomma())) / self.bsmVomma()
        return self.sigma



today = date.today()
expDay = date(2016, 12, 16)
T = expDay - today
sigma0 = 0.5
r = 0.02 # 3-month T-bill rate
currentStockPrice = 16.26


bacOptionList = [[currentStockPrice, 16.00, T.days / 365, r , sigma0, 0.77, 'call'],
                 [currentStockPrice, 17.00, T.days / 365, r, sigma0, 0.31, 'call'],
                 [currentStockPrice, 16.00, T.days / 365, r, sigma0, 0.56, 'put'],
                 [currentStockPrice, 17.00, T.days / 365, r, sigma0, 1.12, 'put']]


print('The underlying asset is Bank of America (BAC), current stock price is ${:.2f}, the expiration date is {:%Y-%m-%d}\n'.format(currentStockPrice, expDay))
print('Now, let\'s begin:\n')

time_start = time.clock()
for option in bacOptionList:
    impvol = ImpliedVolatility(option[0], option[1], option[2], option[3], option[4], option[5], option[6]).bsmBisectionVol()
    print('Here is a {0} option.\nThe strike price is ${1:.2f} and option price is ${2:.2f}.\nThe implied volatility is {3:.16%}\n'.format(option[6], option[1], option[5], impvol))
print('The Bisection method takes {:.4f} seconds to run\n'.format(time.clock() - time_start))

time_start = time.clock()
for option in bacOptionList:
    impvol = ImpliedVolatility(option[0], option[1], option[2], option[3], option[4], option[5], option[6]).bsmMullerBisectionVol()
    print('Here is a {0} option.\nThe strike price is ${1:.2f} and option price is ${2:.2f}.\nThe implied volatility is {3:.16%}\n'.format(option[6], option[1], option[5], impvol))
print('The Muller-Bisection method takes {:.4f} seconds to run\n'.format(time.clock() - time_start))

time_start = time.clock()
for option in bacOptionList:
    impvol = ImpliedVolatility(option[0], option[1], option[2], option[3], option[4], option[5], option[6]).bsmNewtonVol()
    print('Here is a {0} option.\nThe strike price is ${1:.2f} and option price is ${2:.2f}.\nThe implied volatility is {3:.16%}\n'.format(option[6], option[1], option[5], impvol))
print('The Newton method takes {:.4f} seconds to run\n'.format(time.clock() - time_start))

time_start = time.clock()
for option in bacOptionList:
    impvol = ImpliedVolatility(option[0], option[1], option[2], option[3], option[4], option[5], option[6]).bsmHalley()
    print('Here is a {0} option.\nThe strike price is ${1:.2f} and option price is ${2:.2f}.\nThe implied volatility is {3:.15%}\n'.format(option[6], option[1], option[5], impvol))
print('The Halley method takes {:.4f} seconds to run\n'.format(time.clock() - time_start))