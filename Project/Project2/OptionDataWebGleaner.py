from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import pandas as pd
from datetime import datetime


def main():
    timeStart = datetime.now()
    tickerList = pd.read_csv('NasdaqStockList.csv', header=None).iloc[:, 0].values.tolist()
    #tickerList = np.random.choice(tickerList, 10, replace=False)
    print(tickerList)
    data = OptionDataWebGleaner().getAll(tickerList)
    if data:
        cleanData = OptionDataWebGleaner().getAllClean(data)
    print('The download is complete. Running time: {:.2f} mins'.format((datetime.now() - timeStart).seconds/60))


class OptionDataWebGleaner():
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

    def __init__(self, ticker=None, dateIndex=0, expir='', money='all', exCode=''):
        self.ticker = ticker
        self.dateIndex = dateIndex
        self.expir = expir
        self.money = money
        self.exCode = exCode

    # get the content of a certain page
    def getUrl(self, page=1):

        url = 'http://www.nasdaq.com/symbol/' + self.ticker + '/option-chain?dateindex=' + str(
            self.dateIndex) + '&expir=' + self.expir + '&money=' + self.money + '&excode=' \
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

    def getGreeks(self):

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

            if not lst:
                return df

            try:

                arr = np.array(lst).reshape((len(lst) // 16, 16))
                dfTemp = pd.DataFrame(arr)
                df = pd.concat([df, dfTemp])

            except ValueError:
                print('greeks error')

        # make sure the index format in options and greeks are the same, for later dateframe merge
        df.iloc[:, 8] = df.iloc[:, 8].astype(float)

        tuples = list(zip(df.iloc[:, 0], df.iloc[:, 8]))

        index = pd.MultiIndex.from_tuples(tuples, names=['ExpDate', 'StrikePrice'])

        df.rename(columns={8: 'StrikePrice'}, inplace=True)

        callGreeks = df.iloc[:, 0:9]
        putGreeks = df.iloc[:, 7:17]

        callGreeks = callGreeks.set_index(index)
        putGreeks = putGreeks.set_index(index)

        callheader = ['ExpDate', 'Delta', 'Gamma', 'Rho', 'Theta', 'Vega', 'IV', 'Ticker', 'StrikePrice']
        putheader = ['Ticker', 'StrikePrice', 'ExpDate', 'Delta', 'Gamma', 'Rho', 'Theta', 'Vega', 'IV']
        callGreeks.columns = callheader
        putGreeks.columns = putheader

        # reorder and trim the columns
        callGreeks = callGreeks[['Ticker', 'Delta', 'Gamma', 'Rho', 'Theta', 'Vega', 'IV']]
        putGreeks = putGreeks[['Ticker', 'Delta', 'Gamma', 'Rho', 'Theta', 'Vega', 'IV']]

        # manage the datatypes
        callGreeks.iloc[:, 1:] = callGreeks.iloc[:, 1:].astype(float)
        putGreeks.iloc[:, 1:] = putGreeks.iloc[:, 1:].astype(float)

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
    def getOptions(self) -> object:
        df = pd.DataFrame()

        for i in range(self.getPage()):

            table = self.getUrl(i + 1).find_all('table')[5]  # the sixth table contains the option information
            content = table.find_all('td')
            lst = [text.text for text in content]

            if not lst:
                return df

            try:
                arr = np.array(lst).reshape((len(lst) // 16, 16))
                dfTemp = pd.DataFrame(arr)
                df = pd.concat([df, dfTemp])

            except ValueError:
                print('options error')


        # make sure the index format in options and greeks are the same, for later dateframe merge
        df.iloc[:, 8] = df.iloc[:, 8].astype(float)

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
        callOptions = callOptions[['Ticker', 'Last', 'Chg', 'Bid', 'Ask', 'Vol', 'Open Int']]
        putOptions = putOptions[['Ticker', 'Last', 'Chg', 'Bid', 'Ask', 'Vol', 'Open Int']]

        # in order to drop empty strike price rows, replace empty strings with NaN
        callOptions['Last'].replace('', np.nan, inplace=True)
        putOptions['Last'].replace('', np.nan, inplace=True)

        # manage the datatypes
        #callOptions.iloc[:, 5] = callOptions.iloc[:, 5].astype(int)
        #putOptions.iloc[:, 5] = putOptions.iloc[:, 5].astype(int)

        # make sure it returns the correct option data for the underlying asset
        callOptions = callOptions[callOptions.Ticker == self.ticker.upper()]
        putOptions = putOptions[putOptions.Ticker == self.ticker.upper()]

        # add cleaned greeks to the option
        greeks = self.getGreeks()
        callOptions = pd.concat([callOptions, greeks[0][['Delta', 'Gamma', 'Rho', 'Theta', 'Vega', 'IV']]], axis=1)
        putOptions = pd.concat([putOptions, greeks[1][['Delta', 'Gamma', 'Rho', 'Theta', 'Vega', 'IV']]], axis=1)

        # add types to the dataframe
        callOptions.insert(1, 'Type', 'Call')
        putOptions.insert(1, 'Type', 'Put')

        options = pd.concat([callOptions, putOptions])

        return options

    # will return call options only
    def getCall(self):
        return self.getOptions()[self.getOptions().Type == 'Call']

    # will return put options only
    def getPut(self):
        return self.getOptions()[self.getOptions().Type == 'Put']


    def getAll(self, tickerList):

        options = {}
        df = pd.DataFrame()
        filename = 'DirtyData.xlsx'
        writer = pd.ExcelWriter(filename)


        for ticker in tickerList:
            self.ticker = ticker

            # find all expiration date
            lastDate = self.getUrl().find('div', {'id': 'OptionsChain-dates'})
            lastDateIndex = re.findall(pattern='(?:dateindex=)(\d+)', string=str(lastDate))

            if not lastDateIndex:
                expDate = 0
            else:
                expDate = int(lastDateIndex[-1])
            try:

                for date in range(expDate + 1):
                    self.dateIndex = date
                    dfTemp = self.getOptions()
                    if dfTemp.empty:
                        break
                    else:
                        df = pd.concat([df, dfTemp])
            except KeyError:
                continue
            except IndexError:
                continue

            if not df.empty:
                df.to_excel(writer, ticker)
                options[ticker] = df

        return options

    def getAllClean(self, data, N=2000):
        options = {}
        filename = 'CleanData.xlsx'
        writer = pd.ExcelWriter(filename)
        tickerList = np.random.choice(list(data.keys()), min(N, len(list(data.keys()),)), replace=False)

        for ticker in tickerList:
            try:
                df = data[ticker]
                # clean the outliers and missing values
                df = df.loc[(df['IV'] > 0.00) & (df['IV'] < 2.00)]
                #df = df.loc[df['Vol'] > 0]
                df = df.dropna(subset=['Last'])
                df.to_excel(writer, ticker)
            except:
                continue

            options[ticker] = df
        return options

if __name__ == '__main__':
    main()