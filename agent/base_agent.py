import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import abc

import torch
import torch.nn as nn
from torch import optim
from gym.wrappers import Monitor

from .model import MLPAutocorrModel

def make_seed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

class BaseAgent:

    def __init__(self, config):
        self.config = config
        self.env = config['env']
        make_seed(config['seed'])
        self.env.seed(config['seed'])
        self.use_cuda = config['use_cuda']
        self.gamma = config['gamma']
        self.verbose = config['verbose']
        self.max_episode_length = config['max_episode_length']
        self.use_mean_baseline = config.get('use_mean_baseline', False)

        self.model = config['model']

        # the optimizer used by PyTorch (Stochastic Gradient, Adagrad, Adam, etc.)
        self.optimizer = torch.optim.Adam(self.model.net.parameters(), lr=config['learning_rate'])
        self.monitor_env = Monitor(self.env, "./gym-results", force=True, video_callable=lambda episode: True)

    @abc.abstractmethod
    def _compute_returns(self, rewards):
        """Returns the cumulative discounted rewards at each time step

        Parameters
        ----------
        rewards : array
            The array of rewards of one episode

        Returns
        -------
        array
            The cumulative discounted rewards at each time step

        Example
        -------
        for rewards=[1, 2, 3] this method outputs [1 + 2 * gamma + 3 * gamma**2, 2 + 3 * gamma, 3]
        """

        raise NotImplementedError

    def sample_trajectories(self, n_trajectories):
        trajectories = []
        for _ in range(n_trajectories):
            states = [torch.from_numpy(self.env.reset()).type(self.model.dtype)]
            actions = []
            rewards = []
            log_probs = []

            done = False
            count = 0

            # stop after self.max_episode_length steps,
            # otherwise episodes run for too long when
            # the agent is skilled enough
            while not done and count < self.max_episode_length:
                action = int(self.model.select_action(states[-1]))

                prob = self.model.forward(states[-1])
                # clip prob away from 0 and 1 to avoid numerical issues when taking the log
                prob = torch.clamp(prob, np.finfo(np.float32).eps, 1-np.finfo(np.float32).eps)
                log_prob = torch.log(prob)

                state, reward, done, _ = self.env.step(action)
                states.append(torch.from_numpy(state).type(self.model.dtype))
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)

                count += 1

            trajectories.append({
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'log_probs': log_probs,
            })

        return trajectories

    @abc.abstractmethod
    def optimize_model(self, n_trajectories):
        """Perform a gradient update using n_trajectories

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expectation card(D) in the formula above

        Returns
        -------
        array
            The cumulative discounted rewards of each trajectory
        """
        raise NotImplementedError

    def train(self, n_trajectories, n_update):
        """Training method

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expected gradient
        n_update : int
            The number of gradient updates

        """

        rewards = []
        for episode in range(n_update):
            rewards.append(self.optimize_model(n_trajectories))
            if (episode + 1) % self.verbose == 0:
                rewards_np = np.array(rewards)
                # Print the reward stats averaged across all last self.verbose steps
                mean = rewards_np[-1-self.verbose:-1].mean()
                std = rewards_np[-1-self.verbose:-1].std()
                print(f'Episode {episode + 1}/{n_update}: rewards {round(mean, 2)} +/- {round(std, 2)}')

        # Plotting
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards[i]) for i in range(len(rewards))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd');

    def evaluate(self, render=False):
        """Evaluate the agent on a single trajectory
        """

        observation = self.monitor_env.reset()
        observation = torch.tensor(observation, dtype=torch.float)
        reward_episode = 0
        done = False

        while not done:
            action = self.model.select_action(observation)
            observation, reward, done, info = self.monitor_env.step(int(action))
            observation = torch.tensor(observation, dtype=torch.float)
            reward_episode += reward

        self.monitor_env.close()
        if render:
            self.env.render()
        print(f'Reward: {reward_episode}')
