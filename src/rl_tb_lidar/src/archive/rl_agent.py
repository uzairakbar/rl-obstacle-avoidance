#! /usr/bin/env python
import numpy as np
from lvfa import LVFA
from  qlearn import QLearn
from policies import Policy

class Agent():
    def __init__(self, policy="greedy", lvfa=False, dim=7 ,nA=7, nS=78125, epsilon=0.05, alpha=0.01 ,gamma=0.9, ellgibility_trace = True, Q = None ):

        self.policy_class = Policy(nA=nA, epsilon=epsilon)


        if policy == "greedy":
            self.policy = self.policy_class.greedy_policy

        if policy == "eps_greedy":
            self.policy = self.policy_class.eps_policy

        if policy == "softmax":
            self.policy = self.policy_class.softmax_policy

        self.reset_ellgibility_trace = self.do_nothing

        if lvfa:
            self.agent = LVFA(dim=dim, nA=nA, nS=nS,alpha=alpha, gamma=gamma, policy=self.policy)
            self.learn = self.agent.learn
            self.chooseAction = self.agent.chooseAction
            self.save_model = self.agent.save_model
            self.load_model = self.agent.save_model
        else:
            self.agent = QLearn(nA=nA, nS=nS,epsilon=epsilon, alpha=alpha, gamma=gamma, Q=Q, policy=self.policy)
            self.chooseAction = self.agent.chooseAction
            self.save_model = self.agent.save_model
            self.load_model = self.agent.save_model

            if ellgibility_trace:
                self.learn = self.agent.learn_ellgibility_trace
                self.reset_ellgibility_trace = self.agent.reset_ellgibility_trace
            else:
                self.learn = self.agent.learn


    def do_nothing(self):
        pass






