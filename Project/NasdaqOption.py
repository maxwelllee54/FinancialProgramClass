from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import pandas as pd


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

    def __init__(self, ticker, dateIndex = 0, expir = '', money = '', callput = '', exCode = ''):
        self.ticker = ticker
        self.dateIndex = dateIndex
        self.expir = expir
        self.money = money
        self.callput = callput
        self.exCode = exCode

    # get the content of a certain page
    def getUrl(self, page = 1):

        url = 'http://www.nasdaq.com/symbol/' + self.ticker + '/option-chain?dateindex=' + str(self.dateIndex) + '&expir=' + self.expir + '&money=' + self.money + '&callput=' + self.callput + '&excode=' + self.exCode + '&page=' + str(page)

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

            table = self.getUrl(i+1).find_all('table')[5] # the sixth table contains the option information
            content = table.find_all('td')
            lst = [text.text for text in content]

            try:
                arr = np.array(lst).reshape((len(lst)//16, 16))
                dfTemp = pd.DataFrame(arr)
                df = pd.concat([df, dfTemp])

            except ValueError:
                print('error')

        tuples = list(zip(df.iloc[:,0], df.iloc[:, 8] ))

        index = pd.MultiIndex.from_tuples(tuples, names=['Date', 'StrikePrice'])


        df.rename(columns={8: 'StrikePrice'}, inplace=True)
        callOptions = df.iloc[:, 1:7]
        putOptions = df.iloc[:, 10:17]

        callOptions = callOptions.set_index(index)
        putOptions = putOptions.set_index(index)


        header = ['Last', 'Chg', 'Bid', 'Ask', 'Vol', 'Open Int']
        callOptions.columns = header
        putOptions.columns = header

        return callOptions, putOptions

    def getCall(self):
        return self.getAll()[0]

    def getPut(self):
        return self.getAll()[1]

callOption = pd.DataFrame()
putOption = pd.DataFrame()

for i in range(2):
    callOption = pd.concat([callOption, NasdaqOptions('goog', i).getCall()])
    putOption = pd.concat([putOption, NasdaqOptions('goog', i).getCall()])

print(callOption)
print(putOption)



#call = NasdaqOptions('goog').getAll()[0]
#index = call.index

#print(call)
#print(index)