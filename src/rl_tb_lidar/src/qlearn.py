#! /usr/bin/env python
import random
import json
from ast import literal_eval
import numpy as np

class QLearn:
    def __init__(self, actions, states, epsilon, alpha, gamma):
        self.Q = np.zeros((len(states), len(actions)))
        self.actions = actions
        self.states = states
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor

    def learn_Q(self, s1, a1, r, s2):
        TD_targer = r + self.gamma * self.Q[s2, :].max()
        TD_err = TD_targer - self.Q[s1, a1]
        self.Q[s1, a1] = self.Q[s1, a1] + self.alpha * TD_err


    def learn_Q_ellgibility_trace(self, s1, a1, r, s2, E):
        TD_targer = r + self.gamma * self.Q[s2, :].max()
        TD_err = TD_targer - self.Q[s1, a1]
        self.Q = self.Q + self.alpha * E * TD_err


    def chooseAction(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.actions))
        else:
            return np.argmax(self.Q[s, :])

    def saveModel(self, name):
        np.save(name, self.Q)

    def loadModel(self, path):
        self.Q = np.load(path)