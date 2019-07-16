#! /usr/bin/env python
import numpy as np


class LVFA:
    def __init__(self, dim = 7, nA = 7, nS = 78125, alpha = 0.01, gamma = 0.9, policy = None):
        self.dim = dim
        self.K = nA * nS
        self.nA = nA
        self.nS = nS
        self.alpha = alpha
        self.gamma = gamma

        self.policy = policy

        # create feature matrix
        self.create_feature_matrix()

        #init theta
        self.appox_Q = np.zeros(dim)

    def create_feature_matrix(self):
        b_voll_rank = False

        while not(b_voll_rank):
            self.feature_matrix = np.random.rand(self.dim,self.K)
            b_voll_rank = np.linalg.matrix_rank(self.feature_matrix) == self.dim

    def learn(self, s1, a1, r, s2):
        a2 = self.policy(self.get_Q(), s2)
        self.appox_Q = self.appox_Q - self.alpha * (self.appox_Q.T * self.feature_matrix[:, s1 * a1] - r - self.gamma * self.appox_Q.T * self.feature_matrix[:, s2 * a2]) * self.feature_matrix[:, s1 * a1]

    def get_Q(self):
        return np.reshape(self.feature_matrix.T.dot(self.appox_Q),(self.nS, self.nA))

    def chooseAction(self, s):
        return self.policy(self.get_Q(), s)

    def save_model(self,path):
        np.save(path+"-theta", self.appox_Q)
        np.save(path+"-featureMatrix", self.feature_matrix)

    def load_model(self,path):
        self.appox_Q = np.load(path+"-theta")
        self.feature_matrix = np.load(path+"-featureMatrix")
