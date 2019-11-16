import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

from statsmodels.graphics.tsaplots import acf

class rl_trading_env(gym.Env):
    """A synthetic stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 dfs,
                 n_lag=10,
                 n_autocorr=5,
                 observe_type='return',
                 init_balance=0,
                 max_steps=250,
                 reward_mode='pnl',
                 ):
        super(rl_trading_env, self).__init__()

        # dataframe with stacked price time series
        self.dfs = dfs
        # observations can be either prices, returns, or autocorrelations
        self.observe_type = observe_type
        # at each step, observe n_lag prices
        self.n_lag = n_lag
        # in autocorr mode, use first n_autocorr values in the autocorrelogram
        # and the last n_autocorr observed returns
        self.n_autocorr = n_autocorr
        # initial cash amount
        self.init_balance = init_balance
        # maximum step in the environment
        self.max_steps = max_steps

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Discrete(3)

        # Prices contains the OHCL values for the last n_lag prices
        if self.observe_type == 'autocorr':
            self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_autocorr*2, 6), dtype=np.float64)
        elif self.observe_type == 'return':
            self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_lag, 6), dtype=np.float64)            
        else:
            self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_lag+1, 6), dtype=np.float64)

        # Either 'pnl' or 'sharpe'
        self.reward_mode = reward_mode

    def _next_observation(self):
        # Get the stock data points for the last self.n_lag days
        if self.observe_type == 'return':
            obs = self.df_inc.iloc[self.current_step-self.n_lag: self.current_step].values.flatten()
        elif self.observe_type == 'price':
            obs = self.df_inc.iloc[self.current_step-self.n_lag: self.current_step].values.flatten()
        elif self.observe_type == 'autocorr':
            ret = self.df_inc.iloc[self.current_step-self.n_lag: self.current_step].values.flatten()
            obs = np.concatenate([acf(ret, fft=True)[1:self.n_autocorr+1], ret[-1:-self.n_autocorr-1:-1]])
        else:
            raise Exception('{} not an allowed observation type.'.format(self.observe_type))

        return obs

    def _take_action(self, action):
        current_price = self.df.iloc[self.current_step].values[0]

        action_type = action
        amount = 1

        if action_type < 1:
            # Buy amount % of balance in shares
            shares_bought = amount
            self.shares_held += shares_bought
            cost = shares_bought * current_price
            self.balance -= cost

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = amount
            self.shares_held -= shares_sold
            gain = shares_sold * current_price
            self.balance += gain

        self.pnl = self.balance + self.shares_held * current_price - self.net_worth
        self.pnl_history.append(self.pnl)
        self.net_worth += self.pnl

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1

        if self.reward_mode == 'pnl' or np.std(self.pnl_history) <= 1e-3:
            reward = self.pnl
        elif self.reward_mode == 'sharpe':
            reward = self.pnl / (np.std(self.pnl_history))
        else:
            raise Exception('{} not an allowed reward mode.'.format(self.reward_mode))

        done = self.current_step > self.max_steps
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = self.init_balance
        self.net_worth = self.init_balance
        self.max_net_worth = self.init_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        self.pnl = 0
        self.pnl_history = []

        self.df = self.dfs.sample(1, axis=1)
        self.df_inc = (self.df-self.df.shift(1)).iloc[1:]

        self.current_step = self.n_lag

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - self.init_balance

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held}')
        print(
            f'Total PnL: {self.net_worth} (Max PnL: {self.max_net_worth})')
