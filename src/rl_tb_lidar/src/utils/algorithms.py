import numpy as np



class RLAlgorithmMixin(object):
	def __init__(self,
				 nS = 4**4,
				 nA = 10,
				 lvfa = False,
				 feature_size = 4**4,
				 eligibility = True,
				 lamda = 0.9,
				 alpha = 0.1,
				 gamma = 0.9):
		self.lvfa = lvfa
		self.eligibility = eligibility
		if self.eligibility:
			self.lamda = lamda
		else:
			self.lamda = 0.0
		self.alpha = alpha
		self.gamma = gamma
		self.nA = nA
		self.actions_onehot = np.eye(self.nA)
		if self.lvfa:
			self.feature_size = feature_size
			self.params = np.zeros(self.feature_size)
		else:
			self.nS = nS
			self.params = np.zeros((self.nS, self.nA))
		self.E = np.zeros_like(self.params)

	def __call__(self, **kwargs):
		return self.update_value_function(**kwargs)

	def reset_eligibility(self):
		self.E = self.E*0.0



class QLearning(RLAlgorithmMixin):
	def __init__(self,
				 nS = 4**4,
				 nA = 10,
				 lvfa = False,
				 feature_size = 4**4,
				 eligibility = True,
				 lamda = 0.9,
				 alpha = 0.1,
				 gamma = 0.9):
		super(QLearning, self).__init__(nS = nS,
										nA = nA,
										lvfa = lvfa,
										feature_size = feature_size,
										eligibility = eligibility,
										lamda = lamda,
										alpha = alpha,
										gamma = gamma)

	def greedy_lvfa_action(self, state):
		state_features = np.vstack([np.asarray(state)]*self.nA)
		state_action_features = np.concatenate((state_features,
										self.actions_onehot), axis=1)
		approx_values = np.matmul(state_action_features, self.params)
		return np.argmax(approx_values)

	def greedy_action(self, state):
		return np.argmax(self.params[state, :])

	def update_value_function(self,
							  state,
							  action,
							  reward,
							  next_state,
							  next_action = None):

		if self.lvfa:
			next_action = self.greedy_lvfa_action(next_state)
			onehot_a = self.actions_onehot[action, :]
			onehot_a_ = self.actions_onehot[next_action, :]
			phi_s_ = np.concatenate((next_state, onehot_a_))
			phi_s = np.concatenate((state, onehot_a))
			td_target = reward + self.gamma * np.multiply(self.params, phi_s_).sum()
			td_error = td_target - np.multiply(self.params, phi_s).sum()
			self.E = self.gamma * self.lamda * self.E + phi_s
			self.params = self.params + self.alpha * td_error * self.E
		else:
			next_action = self.greedy_action(next_state)
			td_target = reward + self.gamma * self.params[next_state, next_action]
			td_error = td_target - self.params[state, action]
			self.E[state, action] = 1.0
			self.params = self.params + self.alpha * td_error * self.E
			self.E = self.lamda * self.gamma * self.E

		if not self.eligibility:
			self.reset_eligibility()



class SARSA(RLAlgorithmMixin):
	def __init__(self,
				 nS = 4**4,
				 nA = 10,
				 lvfa = False,
				 feature_size = 4**4,
				 eligibility = True,
				 lamda = 0.9,
				 alpha = 0.1,
				 gamma = 0.9):
		super(SARSA, self).__init__(nS = nS,
										nA = nA,
										lvfa = lvfa,
										feature_size = feature_size,
										eligibility = eligibility,
										lamda = lamda,
										alpha = alpha,
										gamma = gamma)

	def update_value_function(self,
							  state,
							  action,
							  reward,
							  next_state,
							  next_action):
		if self.lvfa:
			onehot_a = self.actions_onehot[action, :]
			onehot_a_ = self.actions_onehot[next_action, :]
			phi_s_ = np.concatenate((next_state, onehot_a_))
			phi_s = np.concatenate((state, onehot_a))
			td_target = reward + self.gamma * np.multiply(self.params, phi_s_).sum()
			td_error = td_target - np.multiply(self.params, phi_s).sum()
			self.E = self.gamma * self.lamda * self.E + phi_s
			self.params = self.params + self.alpha * td_error * self.E
		else:
			td_target = reward + self.gamma * self.params[next_state, next_action]
			td_error = td_target - self.params[state, action]
			self.E[state, action] = 1.0
			self.params = self.params + self.alpha * td_error * self.E
			self.E = self.lamda * self.gamma * self.E
		if not self.eligibility:
			self.reset_eligibility()

class RLAlgorithm(RLAlgorithmMixin):
	def __init__(self,
				 algorithm = 'qlearning',
				 nS = 4**4,
				 nA = 10,
				 lvfa = False,
				 feature_size = 4**4,
				 eligibility = True,
				 lamda = 0.9,
				 alpha = 0.1,
				 gamma = 0.9,
				 **kwargs):
		super(RLAlgorithm, self).__init__(nS = nS,
										  nA = nA,
										  lvfa = lvfa,
										  feature_size = feature_size,
										  eligibility = eligibility,
										  lamda = lamda,
										  alpha = alpha,
										  gamma = gamma)
		if algorithm == 'qlearning':
			self.algorithm = QLearning(nS = nS,
									   nA = nA,
									   lvfa = lvfa,
									   feature_size = feature_size,
									   eligibility = eligibility,
									   lamda = lamda,
									   alpha = alpha,
									   gamma = gamma)
		elif algorithm == 'sarsa':
			self.algorithm = SARSA(nS = nS,
								   nA = nA,
								   lvfa = lvfa,
								   feature_size = feature_size,
								   eligibility = eligibility,
								   lamda = lamda,
								   alpha = alpha,
								   gamma = gamma)
		else:
			raise ValueError("algorithm param can only be either qlearning or sarsa, but got "+algorithm)

	def update_value_function(self,
							  state,
							  action,
							  reward,
							  next_state,
							  next_action):
		return self.algorithm.update_value_function(state,
													action,
													reward,
													next_state,
													next_action)
