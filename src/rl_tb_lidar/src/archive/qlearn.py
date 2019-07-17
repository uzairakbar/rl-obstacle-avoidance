#! /usr/bin/env python
import numpy as np



class QLearn:
    def __init__(self, nA, nS, epsilon, alpha, gamma, Q = None, policy = None):
        if Q is None:
            self.Q = np.zeros((nS, nA))
        else:
            self.Q = Q

        self.policy = policy
        self.nA = nA
        self.nS = nS

        self.E = np.zeros((nS, nA))

        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor


    def learn(self, s1, a1, r, s2):
        a2 = self.policy(self.Q,s2)

        TD_targer = r + self.gamma * self.Q[s2, a2]
        TD_err = TD_targer - self.Q[s1, a1]
        self.Q[s1, a1] = self.Q[s1, a1] + self.alpha * TD_err


    def learn_ellgibility_trace(self, s1, a1, r, s2):
        a2 = self.policy(self.Q,s2)

        self.E[s1, a1] = 1.0

        TD_targer = r + self.gamma * self.Q[s2, a2]
        TD_err = TD_targer - self.Q[s1, a1]
        self.Q = self.Q + self.alpha * self.E * TD_err

        self.E = self.E * self.gamma

    def reset_ellgibility_trace(self):
        self.E = np.zeros((self.nS, self.nA))


    def chooseAction(self, s):
        return self.policy(self.Q, s)

    def save_model(self, name):
        np.save(name, self.Q)

    def load_model(self, path):
        self.Q = np.load(path)

