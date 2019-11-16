import numpy as np
from .base_agent import BaseAgent

class REINFORCE(BaseAgent):

    def _compute_returns(self, rewards):
        ###
        # Your code here
        ###
        return np.array([np.dot([self.gamma ** k for k in range(len(rewards)-shift)], rewards[shift:]) for shift in range(len(rewards))])

    def optimize_model(self, n_trajectories):
        ###
        # Your code here
        ###

        trajectories = self.sample_trajectories(n_trajectories)
        loss_samples = []
        reward_trajectories = []

        for trajectory in trajectories:
            R = self._compute_returns(trajectory['rewards'])

            if self.use_mean_baseline:
                R_transformed = R - R.mean()
            else:
                R_transformed = R

            log_prob_state_action = [log_prob[action] for action, log_prob in zip(trajectory['actions'], trajectory['log_probs'])]

            loss_samples.append(sum(R_transformed * log_prob_state_action))
            reward_trajectories.append(sum(trajectory['rewards']))

        # negative log as the loss is to be minimized
        loss = -sum(loss_samples)/n_trajectories

        # The following lines take care of the gradient descent step for the variable loss
        # that you need to compute.

        # Discard previous gradients
        self.optimizer.zero_grad()
        # Compute the gradient
        loss.backward()
        # Do the gradient descent step
        self.optimizer.step()
        return np.array(reward_trajectories)
