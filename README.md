# hurst_trading [ON HOLD FOR NOW]
Environment and models for reinforcement learning applied to trading synthetic signals with statistical autocorrelations.

Fractional Brownian motion (fBM) is a variant of standard Brownian motion that replaces the independant Gaussian increments with correlated noise.

The amount of autorcorrelation is controlled by a parameter H called the Hurst exponent. H=0.5 corresponds to the standard Brownian motion, with no statistical correlation between increments; in a trading context, we would say there is no stattiscal arbitrage opportunity. H<0.5 corresponds to negative autocorrelation : a drop in the signal is more likely to be followed by an increase; this is called a mean-reverting signal. H>0.5 corresponds to positive autocorrelation : an increase is more likely to follow another increase; this is called trend-following.

Using the autocorrelogram of a fBM price, one can design an automated trading strategy to exploit the observed autocorrelation. The goal here is to investigate to what extend a Reinforcement Learning agent using the profit and loss metric (PnL) or the Sharpe ratio (ratio expected outperformance and volatility of PnL) as a reward can learn a similar trading strategy in presence of this statistical bias.

## generate_price_paths

Samples multiple realizations of fBM (using https://github.com/sauxpa/stochastic/tree/master/ito_diffusions) and stores them in a pandas.DataFrame.

## env

Implements a gym environment where each episode is the realization of a price trajectory generated with generate_price_paths.

## agent

Implements a REINFORCE agent to optimize PnL or Sharpe ratio in the above environment.

