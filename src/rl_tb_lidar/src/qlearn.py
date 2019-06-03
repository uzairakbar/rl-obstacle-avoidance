#! /usr/bin/env python

import random
import json
from ast import literal_eval
import numpy as np


class QLearn2:
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

    def chooseAction(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.actions))
        else:
            return np.argmax(self.Q[s, :])

    def saveModel(self, name):
        np.save(name, self.Q)

    def loadModel(self, path):
        self.Q = np.load(path)







class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

    def remap_keys(self, mapping):
        return [str(k) + ":" + str(v) for k, v in mapping.iteritems()]

    def saveModel(self, epoch):
        json.dump(self.remap_keys(self.q), open("model%s.txt" % epoch, 'w'))

    def loadModel(self, path):
        self.q = {}
        temp_array = eval(open(path, 'r').read())
        for val in temp_array:
            self.q[literal_eval(val.split(':')[0])] = literal_eval(val.split(':')[1])

