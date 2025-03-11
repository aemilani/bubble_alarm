import ccxt
import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def linear_func(x, m, c):
    return m * x + c


def get_data():
    bitstamp = ccxt.bitstamp()

    timeframe = '1d'

    start_date = pd.Timestamp('2011-08-18')
    end_date = datetime.datetime.now()

    btc = pd.DataFrame()

    while True:
        ohlcv = bitstamp.fetch_ohlcv('BTC/USD', timeframe, limit=1000, since=int(start_date.timestamp() * 1000))

        if len(ohlcv) == 0:
            break

        chunk_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        chunk_df['timestamp'] = pd.to_datetime(chunk_df['timestamp'], unit='ms')
        chunk_df = chunk_df.set_index('timestamp', drop=True)
        chunk_df = chunk_df[chunk_df.index <= end_date]
        btc = pd.concat([btc, chunk_df])

        if chunk_df.index[-1] == end_date:
            break

        start_date = chunk_df.index[-1] + pd.Timedelta(days=1)

    return btc


if __name__ == '__main__':
    btc = get_data()

    # MA divergence risk
    risk_ma = pd.Series(data=0, index=btc.index)
    thrs = []
    for ma_len in range(10, 201, 1):
        diff = (btc['close'] - btc.ta.ema(ma_len)) / btc.ta.ema(ma_len)
        thrs.append((diff.rolling(4 * 365).mean() + 3 * diff.rolling(4 * 365).std()).rolling(200).mean())
        risk_ma += diff
    thr = pd.concat(thrs, axis=1).dropna().mean(axis=1)
    risk_ma = risk_ma / thr
    risk_ma = (risk_ma - risk_ma.min()) / (risk_ma.max() - risk_ma.min())

    # Pi cycle top risk
    pi_ratio = (btc.ta.sma(111) / (btc.ta.sma(350) * 2))

    params = np.array([-8.105722260235973e-05, 1.274431798089917])
    m, c = params

    x_fit = range(pi_ratio.shape[0])
    y_fit = linear_func(x_fit, m, c)
    y_fit = pd.Series(data=y_fit, index=pi_ratio.index)

    risk_pi = 1 - (y_fit - pi_ratio)

    # Cumulative risk
    risk_cum = (risk_ma + risk_pi) / 2

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax[0].plot(risk_cum.dropna())
    ax[0].axhline(0.8, c='r')
    for idx in risk_cum[risk_cum >= 0.8].index:
        ax[1].axvline(idx, c='r', alpha=0.3)
    ax[1].plot(btc['close'].loc[risk_cum.dropna().index])
    ax[1].set_yscale('log')
    plt.show()
