#! /usr/bin/env python
import numpy as np

class Policy:
    def __init__(self,nA = 7 ,epsilon = 0.25):
        self.min_espsilon = epsilon
        self.epsilon = 0.9
        self.nA = nA


    def greedy_policy(self, Q, s):
        return np.argmax(Q[s,:])


    def eps_policy(self,Q,s):
        if self.epsilon >= self.min_espsilon:
            self.epsilon = self.epsilon * 0.9986

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)
        else:
            return np.argmax(Q[s, :])


    def softmax_policy(self,Q,s):
        probabilities = np.exp(Q[s, :]) / sum(np.exp(Q[s, :]))
        elements = range(self.nA)
        return np.random.choice(elements, p=probabilities)
