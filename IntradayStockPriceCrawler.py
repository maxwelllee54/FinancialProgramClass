# This is the web crawler to get intraday stock price from Google Finance

import requests
import pandas as pd
from io import StringIO
from datetime import datetime

def StockPriceCrawler(ticker, interval = 300, periods = 6, unit = 'M'):
    '''

    :param ticker: The symbol of the stock on Google Finance
    :param interval: The interval in seconds; default is 1 min
    :param periods: the number of the periods
    :param unit:    d for day (default)
                    M for month
                    Y for year
    :return: stock price dataframe
    '''

    #url = 'https://www.google.com/finance/getprices?q=NIFTY&x=NSE&i=60&p=6M&f=d,c,o,h,l&df=cpct&auto=1'
    url = 'http://www.google.com/finance/getprices?q={0}&i={1}&p={2}{3}&f=d,o,h,l,c,v&df=cpct&auto=1'.format(
            str(ticker).upper(), str(interval), str(periods), str(unit))

    headers = {'User-Agent':
                   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) '
                   'AppleWebKit/537.36 (KHTML, like Gecko)'
                   'Chrome/39.0.2171.95 '
                   'Safari/537.36'}

    timeStamp = 0

    try:
        r = requests.get(url, headers=headers)

        start_point = [i for i, word in enumerate(r.text.split()[:10]) if word.startswith('a')][0]

        df = pd.read_csv(StringIO('\n'.join(r.text.split()[start_point:])), sep=',', header=None)
        df.columns = ['Seq', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df.dropna(subset=['Close']).reset_index()

        for i in range(len(df)):
            if df.Seq[i].startswith('a'):
                timeStamp = df.Seq[i][1:]
                df.loc[i, 'Date'] = datetime.fromtimestamp(int(timeStamp))
            else:

                flag = int(df.Seq[i])
                df.loc[i, 'Date'] = datetime.fromtimestamp(int(timeStamp) + flag * interval)


        stockData = df.set_index('Date')[['Open', 'High', 'Low', 'Close', 'Volume']]

        return stockData

    except ConnectionError:
        print('Connection error, try again!')


if __name__ == '__main__':
    df = StockPriceCrawler('aapl')
    print(df)