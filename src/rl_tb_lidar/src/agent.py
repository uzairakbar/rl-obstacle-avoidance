import numpy as np
from utils.algorithms import RLAlgorithm as Algorithm
from utils.policies import Policy

class Agent(object):
	def __init__(self,
				 algorithm = 'qlearning',
				 policy = "eps_greedy",
				 nA = 10,
				 nS = 4**4,
				 lvfa = False,
				 feature_size = 13,
				 alpha = 0.01,
				 gamma = 0.99,
				 eligibility = True,
				 episodes = 1000,
				 load_model = None,
				 **kwargs):
		self.algorithm = Algorithm(algorithm = algorithm,
								   nS = nS,
								   nA = nA,
								   lvfa = lvfa,
								   feature_size = feature_size,
								   eligibility = eligibility,
								   alpha = alpha,
								   gamma = gamma,
								   **kwargs)
		self.policy = Policy(policy = policy,
							 episodes = episodes,
							 nA = nA,
							 lvfa = lvfa,
							 **kwargs)

	def learn(self,
			  state,
			  action_idx,
			  reward,
			  next_state,
			  next_action_idx=None):
		self.algorithm.update_value_function(state,
										  	 action_idx,
										  	 reward,
										  	 next_state,
										  	 next_action_idx)

	def action(self, state, episode = None):
		action = self.policy(state = state,
							 episode = episode,
							 params = self.algorithm.algorithm.params)
		return action

	def reset_eligibility(self):
		self.algorithm.reset_eligibility()

	def save_model(self, path):
		np.save(path, self.algorithm.params)