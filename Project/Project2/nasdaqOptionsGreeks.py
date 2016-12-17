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
            - default       :   near the money
            - in            :   in-the-money
            - out           :   out-the-money
            - all           :   all  the money

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
            self.dateIndex) + '&expir=' + self.expir + '&money=' + self.money + '&callput=' + self.callput + '&excode=' \
              + self.exCode + '&page=' + str(page)

        try:
            r = requests.get(url)

            nasdaqPage = BeautifulSoup(r.content, 'html.parser')

            return nasdaqPage

        except ConnectionError:
            print('Connection error, try again!')

    def getGreeksUrl(self, page=1):
        url = 'http://www.nasdaq.com/symbol/' + self.ticker + '/option-chain/greeks?dateindex=' + str(
            self.dateIndex) + '&page=' + str(page)
        try:
            r = requests.get(url)

            greeksPage = BeautifulSoup(r.content, 'html.parser')

            return greeksPage

        except ConnectionError:

            print('Connection error, try again!')

    def getGreeks(self) -> object:

        lastPage = self.getGreeksUrl().find('a', {'id': 'quotes_content_left_lb_LastPage'})
        lastPageNo = re.findall(pattern='(?:page=)(\d+)', string=str(lastPage))
        pageNo = ''.join(lastPageNo)

        if pageNo == '':
            pageNo = 1
        else:
            pageNo = int(pageNo)

        df = pd.DataFrame()

        for i in range(pageNo):
            table = self.getGreeksUrl(i + 1).find_all('table')[5]
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

        # df.rename(columns={8: 'StrikePrice'}, inplace=True)

        callGreeks = df.iloc[:, 0:9]
        putGreeks = df.iloc[:, 7:17]

        callGreeks = callGreeks.set_index(index)
        putGreeks = putGreeks.set_index(index)

        callheader = ['ExpDate', 'Delta', 'Gamma', 'Rho', 'Theta', 'Vega', 'IV', 'Ticker', 'StrikePrice']
        putheader = ['Ticker', 'StrikePrice', 'ExpDate', 'Delta', 'Gamma', 'Rho', 'Theta', 'Vega', 'IV']
        callGreeks.columns = callheader
        putGreeks.columns = putheader
        # reorder the columns
        callOptions = callGreeks[['Ticker', 'ExpDate', 'Delta', 'Gamma', 'Rho', 'Theta', 'Vega', 'IV', 'StrikePrice']]
        putOptions = putGreeks[['Ticker', 'ExpDate', 'Delta', 'Gamma', 'Rho', 'Theta', 'Vega', 'IV', 'StrikePrice']]
        # in order to drop empty strike price rows, replace empty strings with NaN
        callGreeks['IV'].replace('', np.nan, inplace=True)
        putGreeks['IV'].replace('', np.nan, inplace=True)
        # make sure it returns the correct option data for the underlying asset
        return callGreeks[callGreeks.Ticker == self.ticker.upper()], putGreeks[
            putGreeks.Ticker == self.ticker.upper()]

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
    @property
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

        callOptions = df.iloc[:, 0:9]
        putOptions = df.iloc[:, 7:17]

        callOptions = callOptions.set_index(index)
        putOptions = putOptions.set_index(index)

        callheader = ['ExpDate', 'Last', 'Chg', 'Bid', 'Ask', 'Vol', 'Open Int', 'Ticker', 'StrikePrice']
        putheader = ['Ticker', 'StrikePrice', 'ExpDate', 'Last', 'Chg', 'Bid', 'Ask', 'Vol', 'Open Int']
        callOptions.columns = callheader
        putOptions.columns = putheader
        # reorder the columns
        callOptions = callOptions[['Ticker', 'ExpDate', 'Last', 'Chg', 'Bid', 'Ask', 'Vol', 'Open Int', 'StrikePrice']]
        putOptions = putOptions[['Ticker', 'ExpDate', 'Last', 'Chg', 'Bid', 'Ask', 'Vol', 'Open Int', 'StrikePrice']]
        # in order to drop empty strike price rows, replace empty strings with NaN
        callOptions['Last'].replace('', np.nan, inplace=True)
        putOptions['Last'].replace('', np.nan, inplace=True)
        # make sure it returns the correct option data for the underlying asset
        return callOptions[callOptions.Ticker == self.ticker.upper()], putOptions[
            putOptions.Ticker == self.ticker.upper()]

    # will return a cleaned call dateframe; delete all NaN values
    def getCall(self):
        return self.getAll()[0].dropna(subset=['Last'])

    # will return a cleaned put dataframe; delete all NaN values
    def getPut(self):
        return self.getAll()[1].dropna(subset=['Last'])


for i in range(1):
    df = NasdaqOptions('aapl', i).getGreeks()
    print(df)