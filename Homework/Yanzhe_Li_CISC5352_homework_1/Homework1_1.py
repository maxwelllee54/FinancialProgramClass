# Author: Yanzhe Li
# Date: 10/01/2016
# Description: Evaluate IT and Bank stock risk from volatility

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pylab as py
import datetime
import os.path

def mySkewness(df):
    n = len(df)
    sum = 0
    for i in df:
        sum += (i - np.mean(df)) ** 3
    return (n / ((n-1) * (n-2))) * sum / (np.std(df) ** 3)

def myKurtosis(df):
    n = len(df)
    sum = 0
    for i in df:
        sum += (i - np.mean(df)) ** 4
    return (1 / n) * sum / (np.std(df) ** 4) - 3

def myVol(df):
    S_i = df
    S_i_minus_1 = df.shift(1)

    df['U_seqence'] = np.log(S_i / S_i_minus_1)

    U_seqence = df['U_seqence']

    return pd.rolling_std(U_seqence, window=252) * np.sqrt(252)


start = datetime.datetime(2006, 6, 1)
end = datetime.datetime(2016, 6, 1)

stockTickerIT = ['GOOG', 'YHOO', 'AAPL', 'MSFT', 'AMZN']
stockTickerBank = ['JPM', 'BAC', 'NEW', 'DB', 'C']
stockTickerAll = stockTickerIT + stockTickerBank
statsColumn = ['mean', 'median', 'std', 'skewness', 'kurtosis']
stockData = {}

for ticker in stockTickerIT:
    stockData[ticker] = web.DataReader(ticker, 'google', start, end)

for ticker in stockTickerBank:
    stockData[ticker] = web.DataReader(ticker, 'google', start, end)

# Compare the stock price patterns of the companies
fig1 = py.figure(figsize = (20,10), dpi = 80)
fig1.suptitle('Stock Price Comparison Between IT and Bank Industry', fontsize = 20)

for ticker in stockTickerIT:
    itPlot = py.subplot(2,1,1)
    itPlot.plot(stockData[ticker]['Close'], label = ticker)
    itPlot.legend(loc = 'upper left')
    itPlot.set_title("IT Industry", fontsize = 12)
    py.xlabel('Date')
    py.ylabel('Stock Price')

for ticker in stockTickerBank:
    bankPlot = py.subplot(2,1,2)
    bankPlot.plot(stockData[ticker]['Close'], label = ticker)
    bankPlot.legend(loc='upper right')
    bankPlot.set_title('Bank Industry', fontsize = 12)
    bankPlot.set_title("IT Industry", fontsize=12)
    py.xlabel('Date')
    py.ylabel('Stock Price')

fig1.savefig('Stock Price Comparison Between IT and Bank Industry.png')

# Compare the mean, median, standard deviation, skewness, kurtosis for close price and volume for each data set
stockdf = pd.DataFrame(index = stockTickerAll, columns = statsColumn)

for ticker in stockData:
    stockdf['mean'][ticker] = float(stockData[ticker]['Close'].mean())
    stockdf['median'][ticker] = float(stockData[ticker]['Close'].median())
    stockdf['std'][ticker] = float(stockData[ticker]['Close'].std())
    stockdf['skewness'][ticker] = mySkewness(stockData[ticker]['Close'])
    #stockdf['skewness'][ticker] = stockData[ticker]['Close'].skew()
    stockdf['kurtosis'][ticker] = myKurtosis(stockData[ticker]['Close'])
    #stockdf['kurtosis'][ticker] = stockData[ticker]['Close'].kurtosis()

volumedf = pd.DataFrame(index = stockTickerAll, columns = statsColumn)

for ticker in stockData:
    volumedf['mean'][ticker] = float(stockData[ticker]['Volume'].mean())
    volumedf['median'][ticker] = float(stockData[ticker]['Volume'].median())
    volumedf['std'][ticker] = float(stockData[ticker]['Volume'].std())
    volumedf['skewness'][ticker] = mySkewness(stockData[ticker]['Volume'])
    volumedf['kurtosis'][ticker] = myKurtosis(stockData[ticker]['Volume'])


statsIT1 = stockdf.loc[stockTickerIT, ['mean', 'median', 'std']].mean()
statsBank1 = stockdf.loc[stockTickerBank, ['mean', 'median', 'std']].mean()

statsIT2 = stockdf.loc[stockTickerIT, ['skewness', 'kurtosis']].mean()
statsBank2 = stockdf.loc[stockTickerBank, ['skewness', 'kurtosis']].mean()

statsIT3 = volumedf.loc[stockTickerIT, ['mean', 'median', 'std']].mean()
statsBank3 = volumedf.loc[stockTickerBank, ['mean', 'median', 'std']].mean()

statsIT4 = volumedf.loc[stockTickerIT, ['skewness', 'kurtosis']].mean()
statsBank4 = volumedf.loc[stockTickerBank, ['skewness', 'kurtosis']].mean()

barWidth = 0.3

fig2 = py.figure(figsize = (20,10), dpi = 80)
fig2.suptitle('Stock Price Statistics Data Comparison Between IT and Bank Industry', fontsize = 20)

plot1 = py.subplot(2,2,1)
plot1.bar(np.arange(len(statsIT1)), statsIT1, barWidth, facecolor = '#9999ff', edgecolor='white', label='IT Industry')
plot1.bar(np.arange(len(statsBank1)) + barWidth, statsBank1, barWidth, facecolor = '#ff9999', edgecolor='white', label='Bank Industry')
plot1.set_title("Mean, Median, and Standard Deviation for the Close Price", fontsize = 12)
plot1.legend()
py.ylabel('Value')
py.xticks(np.arange(len(statsIT1)) + barWidth, ('Mean', 'Median', 'Standard Deviation'))


plot2 = py.subplot(2,2,3)
plot2.bar(np.arange(len(statsIT2)), statsIT2, barWidth, facecolor = '#9999ff', edgecolor='white', label='IT Industry')
plot2.bar(np.arange(len(statsBank2)) + barWidth, statsBank2, barWidth, facecolor = '#ff9999', edgecolor='white', label='Bank Industry')
plot2.set_title("Skewness and Kurtosis for the Close Price", fontsize = 12)
plot2.legend()
py.ylabel('Value')
py.xticks(np.arange(len(statsIT1)) + barWidth, ('Skewness', 'Kurtosis'))

plot3 = py.subplot(2,2,2)
plot3.bar(np.arange(len(statsIT3)), statsIT3, barWidth, facecolor = '#9999ff', edgecolor='white', label='IT Industry')
plot3.bar(np.arange(len(statsBank3)) + barWidth, statsBank3, barWidth, facecolor = '#ff9999', edgecolor='white', label='Bank Industry')
plot3.set_title("Mean, Median, and Standard Deviation for the Trading Volume", fontsize = 12)
plot3.legend()
py.ylabel('Value')
py.xticks(np.arange(len(statsIT1)) + barWidth, ('Mean', 'Median', 'Standard Deviation'))

plot4 = py.subplot(2,2,4)
plot4.bar(np.arange(len(statsIT4)), statsIT4, barWidth, facecolor = '#9999ff', edgecolor='white', label='IT Industry')
plot4.bar(np.arange(len(statsBank4)) + barWidth, statsBank4, barWidth, facecolor = '#ff9999', edgecolor='white', label='Bank Industry')
plot4.set_title("Skewness and Kurtosis for the Trading Volume", fontsize = 12)
plot4.legend()
py.ylabel('Value')
py.xticks(np.arange(len(statsIT1)) + barWidth, ('Skewness', 'Kurtosis'))

fig2.savefig('Stock Price Statistics Data Comparison Between IT and Bank Industry.png')


# Output all stock prices (close price) which are >= 95% percentile and their corresponding volumes and dates for each data set

newStockData = {}

for ticker in stockTickerAll:
    newStockData[ticker] = stockData[ticker].loc[stockData[ticker]['Close'] > stockData[ticker]['Close'].quantile(0.95)]


for ticker in stockTickerAll:
    stockData[ticker]['Volatility'] = myVol(stockData[ticker]['Close'])

fig3 = py.figure(figsize = (20,10), dpi = 80)
fig3.suptitle('Stock Volatility Comparison Between IT and Bank Industry', fontsize = 20)

for ticker in stockTickerIT:
    itPlot = py.subplot(2,1,1)
    itPlot.plot(stockData[ticker]['Volatility'], label = ticker)
    itPlot.legend(loc = 'upper left')
    itPlot.set_title("IT Industry", fontsize = 12)
    py.xlabel('Date')
    py.ylabel('Volatility')

for ticker in stockTickerBank:
    bankPlot = py.subplot(2,1,2)
    bankPlot.plot(stockData[ticker]['Volatility'], label = ticker)
    bankPlot.legend(loc='upper left')
    bankPlot.set_title('Bank Industry', fontsize = 12)
    py.xlabel('Date')
    py.ylabel('Volatility')

fig3.savefig('Stock Volatility Comparison Between IT and Bank Industry')

# save the stock data into excel
filename = ('StockData.xlsx')

writer = pd.ExcelWriter(filename)
for ticker in stockData:
    stockData[ticker].to_excel(writer, ticker)
    newStockData[ticker][['Close','Volume']].to_excel(writer, ticker + '95')

writer.save()

if (os.path.isfile(filename) == True):
    print(filename + ' is saved')