import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sensormodel import Hit, Short, Max, Rand, Likelihood, SampleLIDAR

ITERATIONS = 25

def ExpectationMaximization(theta,
                            measurements,
                            ground_truth,
                            iterations=ITERATIONS,
                            num_features=360):

    cols = ['lidar'+str(i) for i in range(num_features)]

    P_HIT = measurements[cols].values.copy()
    P_SHORT = measurements[cols].values.copy()
    P_MAX = measurements[cols].values.copy()
    P_RAND = measurements[cols].values.copy()
    z = measurements[cols].values.copy()
    z_ = measurements[cols].values.copy()

    p_max = np.vectorize(Max().P)
    p_rand = np.vectorize(Rand().P)

    for pos in measurements["pos"].unique():
        pos_mask = (measurements['pos']==pos)
        pos_measurements = measurements[cols][pos_mask]

        pos_ground_truth_row = ground_truth[cols][ground_truth['pos']==pos]
        position_ground_truth = pd.concat([pos_ground_truth_row]*len(pos_measurements), ignore_index=True)

        z_[pos_mask] = position_ground_truth.values

    P_MAX = p_max(z, z_)
    P_RAND = p_rand(z, z_)
    P_MAX_RAND = P_MAX + P_RAND

    likelihood_log = []

    for i in range(ITERATIONS):
        p_hit = np.vectorize(Hit(sigma = theta[-2]).P)
        p_short = np.vectorize(Short(lamda = theta[-1]).P)

        P_HIT = p_hit(z, z_)
        P_SHORT = p_short(z, z_)

        eta = np.power(P_HIT + P_SHORT + P_MAX_RAND, -1)

        e_hit = np.multiply(eta, P_HIT)
        e_short = np.multiply(eta, P_SHORT)
        e_max = np.multiply(eta, P_MAX)
        e_rand = np.multiply(eta, P_RAND)

        z_hit = e_hit.sum()*1.0/e_hit.size
        z_short = e_short.sum()*1.0/e_short.size
        z_max = e_max.sum()*1.0/e_max.size
        z_rand = e_rand.sum()*1.0/e_rand.size

        x = np.multiply(e_hit, np.square(z - z_))
        sigma_hit = np.sqrt(x.sum()/e_hit.sum())

        lamda_short = e_short.sum()*1.0/np.multiply(e_short, z).sum()

        theta = np.asarray([z_hit, z_short, z_max, z_rand, sigma_hit, lamda_short])
        print("iteration: ", i)
        likelihood = np.vectorize(Likelihood(theta = theta).P)
        likelihood_log.append(likelihood(z, z_).sum()/z.size)
        print("likelihood: ", likelihood_log[-1])
        print("params: ", theta)
    fig, ax = plt.subplots()
    ax.plot(likelihood_log)
    ax.grid()
    return theta, likelihood_log, fig, ax
