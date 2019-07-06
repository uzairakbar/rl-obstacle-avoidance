import numpy as np

class Policy:
    def __init__(self,nA = 7 ,epsilon = 0.25):
        self.epsilon = epsilon
        self.nA = nA


    def greedy_policy(self, Q, s):
        return np.argmax(Q[s,:])


    def eps_policy(self,Q,s):

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)
        else:
            return np.argmax(Q[s, :])


    def softmax_policy(self,Q,s):
        probabilities = np.exp(Q[s, :]) / sum(np.exp(Q[s, :]))
        elements = range(self.nA)
        return np.random.choice(elements, p=probabilities)
