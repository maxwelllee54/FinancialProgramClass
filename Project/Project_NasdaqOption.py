from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import pandas as pd
import math as e
from scipy import stats
from datetime import date, timedelta, datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class NasdaqOptions():
    '''
        Let user to scrape option data from NASDAQ option chain:

        Input:
        ticker:     ticker for the underlying stock
        dateindex:  expiration date
            - 0(default): current month
            - 1         : 1st nearby month
            - 2         : 2nd nearby month
            ....
        expir:      expiration gap
            - default       :   all
            - stan          :   monthly
            - week          :   weekly
            - quart         :   quarterly
            ...

        money:  strick price's moneyness
            - default       :   all
            - in            :   in-the-money
            - out           :   out-the-money
            - near          :   near the money

        callput:    call or put options
            - default       :   all
            - call          :   call option
            - put           :   put option
            ...

        excode:     option exchange
            - default       :   all
            - composite     :   composite quote
            - cbo           :   Chicago Board Options Exchange
            - aoe           :   American Options Exchange
            - nyo           :   New York Options Exchange
            - pho           :   Philadelphia Options Exchange
            - moe           :   Montreal Options Exchange
            - box           :   Boston Options Exchange
            - ise           :   International Securities Exchange
            - bto           :   Bats Exchange Options Market
            - nso           :   NASDAQ Options
            - c2o           :   C2(Chicago) Options Exchange
            - bxo           :   Bats Exchange Options Market
            - miax          :   MIAX
        '''

    def __init__(self, ticker, dateIndex=0, expir='', money='', callput='', exCode=''):
        self.ticker = ticker
        self.dateIndex = dateIndex
        self.expir = expir
        self.money = money
        self.callput = callput
        self.exCode = exCode

    # get the content of a certain page
    def getUrl(self, page=1):

        url = 'http://www.nasdaq.com/symbol/' + self.ticker + '/option-chain?dateindex=' + str(
            self.dateIndex) + '&expir=' + self.expir + '&money=' + self.money + '&callput=' + self.callput + '&excode=' + self.exCode + '&page=' + str(
            page)

        try:
            r = requests.get(url)

            nasdaqPage = BeautifulSoup(r.content, 'html.parser')

            return nasdaqPage

        except ConnectionError:
            print('Connection error, try again!')

    # get the total page number
    def getPage(self):

        lastPage = self.getUrl().find('a', {'id': 'quotes_content_left_lb_LastPage'})
        lastPageNo = re.findall(pattern='(?:page=)(\d+)', string=str(lastPage))
        pageNo = ''.join(lastPageNo)

        if pageNo == '':
            return 1
        else:
            return int(pageNo)

    # get all pages' option information and put into DataFrame
    def getAll(self):
        df = pd.DataFrame()

        for i in range(self.getPage()):

            table = self.getUrl(i + 1).find_all('table')[5]  # the sixth table contains the option information
            content = table.find_all('td')
            lst = [text.text for text in content]

            try:
                arr = np.array(lst).reshape((len(lst) // 16, 16))
                dfTemp = pd.DataFrame(arr)
                df = pd.concat([df, dfTemp])

            except ValueError:
                print('error')

        tuples = list(zip(df.iloc[:, 0], df.iloc[:, 8]))

        index = pd.MultiIndex.from_tuples(tuples, names=['ExpDate', 'StrikePrice'])

        df.rename(columns={8: 'StrikePrice'}, inplace=True)
        callOptions = df.iloc[:, 0:8]
        putOptions = df.iloc[:, 8:17]

        callOptions = callOptions.set_index(index)
        putOptions = putOptions.set_index(index)

        callheader = ['ExpDate', 'Last', 'Chg', 'Bid', 'Ask', 'Vol', 'Open Int', 'StrikePrice']
        putheader = ['StrikePrice', 'ExpDate', 'Last', 'Chg', 'Bid', 'Ask', 'Vol', 'Open Int']
        callOptions.columns = callheader
        putOptions.columns = putheader

        return callOptions, putOptions

    def getCall(self):
        return self.getAll()[0]

    def getPut(self):
        return self.getAll()[1]


class ImpliedVolatility():
    def __init__(self, S, K, T, r, sigma, cStar, optionType, err=1e-5):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.cStar = cStar
        self.optionType = optionType
        self.err = float(err)

    def bsmValue(self, sigma=None):
        if sigma == None:
            sigma = self.sigma

        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)

        if self.optionType in ['Call', 'call', 'CALL']:

            return self.S * stats.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2)

        elif self.optionType in ['Put', 'put', 'PUT']:

            return self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2) - self.S * stats.norm.cdf(-d1)

        else:
            raise TypeError('the option_type argument must be either "call" or "put"')

    def bsmVega(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        vega = self.S * stats.norm.pdf(d1) * np.sqrt(self.T)
        return vega

    def bsmVomma(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        return self.bsmVega() * d1 * d2 / self.sigma

    def f(self, sigma=None):
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
        middle = (lower + upper) / 2
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
            self.sigma = self.sigma - self.f() / self.bsmVega()
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
            self.sigma = self.sigma + (-self.bsmVega() + e.sqrt(
                self.bsmVega() ** 2 - 2 * self.f() * self.bsmVomma())) / self.bsmVomma()
        return self.sigma


if __name__ == '__main__':
    # tickerList = ['GOOG', 'YHOO', 'AAPL', 'MSFT', 'JPM', 'BAC', 'HSBC', 'CIT', 'GS']
    tickerList = ['GOOG']
    dateDic = {'Dec 2016': 2}
    # dateDic = {'Dec 2016': 2, 'Jan 2017': 3, 'Feb 2017': 4}
    callOptionDic = {}
    putOptionDic = {}
    today = datetime.today()
    sigma0 = 0.5
    r = 0.02  # 3-month T-bill rate

    for ticker in tickerList:
        callOptionDic[ticker] = pd.DataFrame()
        putOptionDic[ticker] = pd.DataFrame()

        for date in dateDic:
            callOptionDic[ticker] = pd.concat(
                [callOptionDic[ticker], NasdaqOptions(ticker=ticker, dateIndex=dateDic[date]).getCall()])
            callOptionDic[ticker]['StockPrice'] = float(
                web.DataReader(ticker, 'google', today - timedelta(2), today - timedelta(1))['Close'])

            putOptionDic[ticker] = pd.concat(
                [putOptionDic[ticker], NasdaqOptions(ticker=ticker, dateIndex=dateDic[date]).getPut()])
            putOptionDic[ticker]['StockPrice'] = float(
                web.DataReader(ticker, 'google', today - timedelta(2), today - timedelta(1))['Close'])

    print(callOptionDic)
    print(putOptionDic)

    for ticker in tickerList:

        #callIndex = callOptionDic[ticker].index
        #print(callIndex)

        #for index in callIndex:
            try:
                S = callOptionDic[ticker].ix[:, 'StockPrice']
                K = callOptionDic[ticker].ix[:, 'StrikePrice']
                for date in callOptionDic[ticker].ix[:, 'ExpDate']:

                T = datetime.strptime(str, '%b %d, %Y') - today
                cStar = callOptionDic[ticker].ix[index, 'Last']
                if cStar is not None:
                try:
                    cStar = float(cStar)
                    callOptionDic[ticker].ix[index, 'ImpVolBisec'] = ImpliedVolatility(S, K, T.days / 365.0, r,
                                                                                       sigma0, cStar,
                                                                                       'call').bsmBisectionVol()

                    #callOptionDic[ticker].ix[index, 'MullerBisec'] = ImpliedVolatility(S, K,
                    #                                                                   T.days / 365.0, r,
                    #                                                                   sigma0, cStar,
                    #                                                                   'call').bsmMullerBisectionVol()

                    #callOptionDic[ticker].ix[index, 'Newton'] = ImpliedVolatility(S, K, T.days / 365.0, r,
                    #                                                             sigma0, cStar,
                    #                                                              'call').bsmNewtonVol()

                    #callOptionDic[ticker].ix[index, 'Halley'] = ImpliedVolatility(S, K, T.days / 365.0, r,
                    #                                                              sigma0, cStar,
                    #                                                              'call').bsmHalley()
                except ValueError:
                    pass
            print('\nok\n')


        putIndex = putOptionDic[ticker].index

        for index in putIndex:
            S = float(putOptionDic[ticker].ix[index, 'StockPrice'])
            K = float(index[1])
            str = index[0]
            T = datetime.strptime(str, '%b %d, %Y') - today
            cStar = putOptionDic[ticker].ix[index, 'Last']
            if cStar is not None:
                try:
                    cStar = float(cStar)
                    putOptionDic[ticker].ix[index, 'ImpVolBisec'] = ImpliedVolatility(S, K, T.days / 365.0, r,
                                                                                      sigma0, cStar,
                                                                                      'call').bsmBisectionVol()

                    #putOptionDic[ticker].ix[index, 'MullerBisec'] = ImpliedVolatility(S, K, T.days / 365.0, r,
                    #                                                                  sigma0, cStar,
                    #                                                                  'call').bsmMullerBisectionVol()

                    #putOptionDic[ticker].ix[index, 'Newton'] = ImpliedVolatility(S, K, T.days / 365.0, r,
                    #                                                             sigma0, cStar,
                    #                                                            'call').bsmNewtonVol()

                    #putOptionDic[ticker].ix[index, 'Halley'] = ImpliedVolatility(S, K, T.days / 365.0, r,
                    #                                                             sigma0, cStar,
                    #                                                             'call').bsmHalley()
                except ValueError:
                    pass

    print(callOptionDic)
    # print(test)



strike_price = 0
time_to_maturity = np.linspace(0.25, 3, 50)
strike_price, time_to_maturity = np.meshgrid(strike_price, time_to_maturity)

fig = plot.figure(figsize=(10, 5))
ax = Axes3D(fig)

'''
today = date.today()
expDay = date(2016, 12, 16)
T = expDay - today

sigma0 = 0.5
r = 0.02 # 3-month T-bill rate
currentStockPrice = 16.1


bacOptionList = [[currentStockPrice, 16.00, T.days / 365, r , sigma0, 0.80, 'call'],
                 [currentStockPrice, 17.00, T.days / 365, r, sigma0, 0.38, 'call'],
                 [currentStockPrice, 16.00, T.days / 365, r, sigma0, 0.74, 'put'],
                 [currentStockPrice, 17.00, T.days / 365, r, sigma0, 1.32, 'put']]

print('The underlying asset is Bank of America (BAC), current stock price is ${:.2f}, the expiration date is {:%Y-%m-%d}\n'.format(currentStockPrice, expDay))
print('Now, let\'s begin:\n')

for option in bacOptionList:
    im = ImpliedVolatility(option[0], option[1], option[2], option[3], option[4], option[5], option[6])
    newton = im.bsmNewtonVol()
    bisection = im.bsmBisectionVol()
    mullerBisection = im.bsmMullerBisectionVol()
    halley = im.bsmHalley()
    print('Here is a {0} option.\nThe strike price is ${1:.2f} and option price is ${2:.2f}.\nThe implied volatility is {3:.2%}\n'.format(option[6], option[1], option[5], newton))

    print(bisection)
    print(mullerBisection)
    print(halley)

strikePrice = list(callOptionDic[ticker].ix[:, 'StrikePrice'])
        timeToMaturity = list(callOptionDic[ticker].ix[:, 'TimeToMaturity'])
        impVol = list(callOptionDic[ticker].ix[:, 'ImpVol'])
        print(strikePrice)
        strikePrice, timeToMaturity = np.meshgrid(strikePrice, timeToMaturity)

        fig = plot.figure(figsize=(10, 5))
        ax = Axes3D(fig)
        surf = ax.plot_surface(strikePrice, timeToMaturity, impVol, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0.5, antialiased=False)

        #ax.set_xlable('Strike Price')
        #ax.set_ylable('Time to Maturity')
        #ax.set_zlabel('Implied Volatility')
'''
