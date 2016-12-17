import numpy as np
from scipy import stats
import pandas as pd
import pylab as py
import os.path


# test multiple barrier values and sigmas
def main():
    Sb = [1] + [x for x in range(10,80,10)]
    Sigma = [x for x in np.arange(0.1, 0.6, 0.05)]

    df1 = pd.DataFrame(index=Sb, columns=['Barrier_Option_Price', 'BS_Option_Price'])
    for sb in Sb:
        df1.loc[sb, 'Barrier_Option_Price'] = bbs_put_value(100, 105, sb, 0.75, 0.05, 0.4)
        df1.loc[sb, 'BS_Option_Price'] = bs_put_value(100, 105, 0.75, 0.05, 0.4)

    df2 = pd.DataFrame(index=Sigma, columns=['Barrier_Option_Price'])
    for sigma in Sigma:
        df2.loc[sigma, 'Barrier_Option_Price'] = bbs_put_value(100, 105, 60, 0.75, 0.05, sigma)

    fig = py.figure(figsize=(20, 10), dpi=80)
    fig.suptitle('Barrier Option Price vs Volatility', fontsize=20)
    ax = py.subplot()
    ax.plot(df2, color="green", linewidth=1.0, linestyle="-", label = 'Barrier Option Price')
    ax.legend(loc='upper left')
    py.xlabel('Volatility')
    py.ylabel('Barrier Option Price')

    fig.savefig('Barrier Option Price vs Volatility.png')


    filename = ('BarrierOptionPrice.xlsx')

    writer = pd.ExcelWriter(filename)

    df1.to_excel(writer, 'Barrier_VS_BSModel')
    df2.to_excel(writer, 'Barrier_VS_Volatility')

    writer.save()

    if (os.path.isfile(filename) == True):
        print(filename + ' is saved')


def bbs_put_value(S, K, Sb, T, r, sigma):

    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S/K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d3 = (np.log(S/Sb) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d4 = (np.log(S/Sb) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d5 = (np.log(S/Sb) - (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d6 = (np.log(S/Sb) - (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d7 = (np.log(S * K / Sb ** 2) - (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d8 = (np.log(S * K / Sb ** 2) - (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    a = (Sb / S) ** (-1 + 2 * r / sigma ** 2)
    b = (Sb / S) ** (1 + 2 * r / sigma ** 2)

    N_d1 = stats.norm.cdf(d1)
    N_d2 = stats.norm.cdf(d2)
    N_d3 = stats.norm.cdf(d3)
    N_d4 = stats.norm.cdf(d4)
    N_d5 = stats.norm.cdf(d5)
    N_d6 = stats.norm.cdf(d6)
    N_d7 = stats.norm.cdf(d7)
    N_d8 = stats.norm.cdf(d8)

    return K * np.exp(- r * T) * (N_d4 - N_d2 - a * (N_d7 - N_d5)) - S * (N_d3 - N_d1 - b * (N_d8 - N_d6))

def bs_put_value(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    N_d1 = stats.norm.cdf(-d1, 0.0, 1.0)
    N_d2 = stats.norm.cdf(-d2, 0.0, 1.0)

    put_price = (-S * N_d1 + K * np.exp(-r * T) * N_d2)

    return put_price


main()