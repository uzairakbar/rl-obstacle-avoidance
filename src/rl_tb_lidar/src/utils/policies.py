import numpy as np



class RandomPolicy(object):
	def __init__(self,
				 nA = 10,
				 lvfa = False):
		self.nA = nA
		self.lvfa = lvfa
		self.actions_onehot = np.eye(self.nA)

	def random_action(self):
		action_idx = np.random.randint(self.nA)
		return action_idx

	def action(self, params=None):
		return self.random_action()

	def __call__(self, params=None):
		return self.action(params)



class GreedyPolicy(RandomPolicy):
	def __init__(self,
				 nA = 10,
				 lvfa = False,
				 **kwargs):
		super(GreedyPolicy, self).__init__(nA = nA,
										   lvfa = lvfa)
	def approx_values(self, state, params):
		state_features = np.vstack([np.asarray(state)]*self.nA)
		state_action_features = np.concatenate((state_features,
										self.actions_onehot), axis=1)
		approx_values = np.matmul(state_action_features, params)
		return approx_values

	def greedy_action(self,
					  state,
					  params,
					  episode=None):
		if self.lvfa == False:
			values = params[state, :]
			# action_idx = np.argmax(params[state, :])
		else:
			values = self.approx_values(state, params)
			# state_features = np.vstack([np.asarray(state)]*self.nA)
			# state_action_features = np.concatenate((state_features,
			# 								self.actions_onehot), axis=1)
			# approx_values = np.matmul(state_action_features, params)
			# action_idx = np.argmax(approx_values)
		action_idx = np.argmax(values)
		return action_idx

	def action(self,
			   state,
			   params,
			   episode=None):
		action_idx = self.greedy_action(state, params, episode)
		return action_idx

	def __call__(self,
				 state,
				 params,
				 episode=None):
		return self.action(state, params, episode)



class EpsilonGreedyPolicy(GreedyPolicy):
	def __init__(self,
				 nA = 10,
				 lvfa = False,
				 epsilon = 0.05,
				 episodes = 1000,
				 **kwargs):
		super(EpsilonGreedyPolicy, self).__init__(nA = nA,
												  lvfa = lvfa)
		self.epsilon = epsilon
		try:
			self.epsilon_list = np.concatenate((np.geomspace(1.0, epsilon, int(episodes*2.0/3)),
											np.repeat(epsilon, episodes - int(episodes*2.0/3))))
		except:
			self.epsilon_list = np.concatenate((np.logspace(np.log10(1.0), np.log10(epsilon), int(episodes*2.0/3)),
											np.repeat(epsilon, episodes - int(episodes*2.0/3))))

	def action(self,
			   state,
			   params,
			   episode=None):
		draw = np.random.random()
		try:
			eps = self.epsilon_list[episode]
		except:
			eps = self.epsilon
		if draw < eps:
			action_idx = self.greedy_action(state, params, episode)
		else:
			action_idx = self.random_action()
		return action_idx



def softmax(x, axis=None):
	x = x - x.max(axis=axis, keepdims=True)
	y = np.exp(x)
	return y/y.sum(axis=axis, keepdims=True)



class SoftmaxPolicy(GreedyPolicy):
	def __init__(self,
				 nA = 10,
				 lvfa = False,
				 temperature = 0.05,
				 episodes = 1000,
				 **kwargs):
		super(SoftmaxPolicy, self).__init__(nA = nA,
												  lvfa = lvfa)
		self.temperature = temperature
		try:
			self.temperature_list = np.concatenate((np.geomspace(1.0, temperature, int(episodes*2.0/3)),
											np.repeat(temperature, episodes - int(episodes*2.0/3))))
		except:
			self.temperature_list = np.concatenate((np.logspace(np.log10(1.0), np.log10(temperature), int(episodes*2.0/3)),
											np.repeat(temperature, episodes - int(episodes*2.0/3))))

	def action(self,
			   state,
			   params,
			   episode=None,
			   **kwargs):
		try:
			temp = self.temperature_list[episode]
		except:
			temp = self.temperature

		if self.lvfa == False:
			logits = params[state, :]/temp
		else:
			# state_features = np.vstack([np.asarray(state)]*self.nA)
			# state_action_features = np.concatenate((state_features,
			# 								self.actions_onehot), axis=1)
			# logits = np.matmul(state_action_features, params)/temp
			logits = self.approx_values(state, params)/temp
		probabilities = softmax(logits)
		action_idx = np.random.choice(np.arange(self.nA), p=probabilities)
		return action_idx



class Policy(GreedyPolicy):
	def __init__(self,
				 policy = 'eps_greedy',
				 nA = 10,
				 lvfa = False,
				 episodes = 1000,
				 **kwargs):
		super(Policy, self).__init__(nA = nA,
									 lvfa = lvfa)
		if policy == 'greedy':
			self.policy = GreedyPolicy(nA = nA,
									   lvfa = lvfa,
									   **kwargs)
		if policy == 'eps_greedy':
			self.policy = EpsilonGreedyPolicy(nA = nA,
											  lvfa = lvfa,
											  episodes = episodes,
											  **kwargs)
		if policy == 'softmax':
			self.policy = SoftmaxPolicy(nA = nA,
										lvfa = lvfa,
										episodes = episodes,
										**kwargs)

	def action(self,
			   state,
			   params,
			   episode=None):
		return self.policy.action(state, params, episode)
