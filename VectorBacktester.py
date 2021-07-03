## Python Module for Vectorised Backtesting

import numpy as np
import pandas as pd
from scipy.optimize import brute
import yfinance as yf

class VectorBacktester(object):
    """ Class for vectorized backtesting

    Attributes
    ==========
    symbol: str
        symbol to work with
    SMA1: int
        time window for shorter MA
    SMA2: int
        time window for longer MA
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    amount: int, float
        amount to be invested at the beginning
    tc: float
        proportional transaction costs (e.g. 0.5% = 0.005) per trade

    Methods
    =======
    get_data:
        retrieves and preprocesses the base dataset
    set_parameters:
        sets the SMA parameters
    run_strategy:
        runs the backtest for SMA-based strategy
    plot_results:
        plots the performance of the strategy
    update_and_run:
        updates the parameters and rerun rhe strategy
    optimize_parameters:
        implements a brute force optimization for the two SMA parameters
    """

    def __init__(self, symbol, SMA1, SMA2, start, end, amount, tc):
        self.symbol = symbol
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.get_data()
        self.results = None

    def get_data(self):
        """
        Retrieves and preprocesses the base dataset
        """
        raw = yf.download(self.symbol, self.start, self.end).dropna()
        data = pd.DataFrame(raw['Adj Close'])
        data.rename(columns={'Adj Close': 'price'}, inplace=True)
        data['return'] = np.log(data / data.shift(1))
        data['SMA1'] = data['price'].rolling(self.SMA1).mean()
        data['SMA2'] = data['price'].rolling(self.SMA2).mean()
        self.data = data

    def set_parameters(self, SMA1=None, SMA2=None):
        """
        Updates SMA Parameters and resp. time series
        """
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data['SMA1'] = self.data['price'].rolling(self.SMA1).mean()

        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data['SMA2'] = self.data['price'].rolling(self.SMA2).mean()

    def run_strategy(self):
        """
        Backtests the trading strategy
        """
        data = self.data.copy().dropna()
        data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        data['strategy'] = data['position'].shift(1) * data['return']
        data.dropna(inplace=True)
        # determine when a trade takes place
        trades = data['position'].diff().fillna(0) != 0
        # subtract transaction costs from return when trade takes place
        data['strategy'][trades] -= self.tc
        data['Benchmark'] = self.amount * data['return'].cumsum().apply(np.exp)
        data['cStrategy'] = self.amount * data['strategy'].cumsum().apply(np.exp)
        self.results = data
        absperf = data['cStrategy'].iloc[-1]
        relperf = absperf - data['Benchmark'].iloc[-1]
        return round(absperf, 2), round(relperf, 2)

    def plot_results(self):
        """
        Plots the cumulative performance of the trading strategy
        """
        if self.results is None:
            print('No results to plot. Run a strategy.')

        title = '%s | SMA1=%d, SMA2=%d' % (self.symbol, self.SMA1, self.SMA2)
        self.results[['Benchmark', 'cStrategy']].plot(title=title, figsize=(10, 6))

    def update_and_run(self, SMA):
        """ Updates SMA parameters and returns negative absolute performance
        (for minimazation algorithm).
        Parameters
        ----------
        SMA:tuple
            SMA Parameter tuple
        Returns
        -------
        negative absolute performance
        """
        self.set_parameters(int(SMA[0]), int(SMA[1]))
        return -self.run_strategy()[0]

    def optimize_parameters(self, SMA1_range, SMA2_range):
        """ Find global maximum given the SMA parameter range
        Parameters
        ----------
        SMA1_range: tuple
        SMA2_range: tuple
            (start, end, step size)
        Returns
        -------
        Optimized parameters, and updated strategy perf
        """
        opt = brute(self.update_and_run, (SMA1_range, SMA2_range), finish=None)
        return opt, -self.update_and_run(opt)

if __name__ == '__main__':
    #smvb = VectorBacktester('EURUSD%3DX', 42, 252, '2010-1-1', '2020-12-31', 10000, 0.0)
    smvb = VectorBacktester('EURUSD%3DX', 42, 252, '2010-1-1', '2020-12-31', 10000, 0.001)
    print(smvb.run_strategy())
    smvb.set_parameters(SMA1=20, SMA2=100)
    print(smvb.run_strategy())
    print(smvb.optimize_parameters((30, 56, 4), (200, 300, 4)))

# (1.58, 0.65)
# (1.08, 0.1)
# (array([ 50., 236.]), 16424.9) # TC = 0
# (array([ 50., 236.]), 16294.03) # TC = 0.001






