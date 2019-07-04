#! /usr/bin/env python
import rospy
import numpy
import random
import time

import numpy as np
import qlearn
import lidar_env as en
import sys
from sensor_msgs.msg import LaserScan


def handle_collision(env, agent, base_filename, episodeRewardLog, last_action):
    print "Saving model and training log with " + base_filename + " as base filename."
    filename = "Qinit_" + base_filename
    agent.saveModel(filename)
    filename = "trainingRewardLog_" + base_filename
    np.save(filename, np.asarray(episodeRewardLog))
    # apple the reserve action to move the turtlebot back.
    return env.handle_collision(last_action)


def run():
    rospy.init_node('rl_agent_tb')
    env = en.Turtlebot_Lidar_Env()
    base_filename = 'Qlearning'
    # Save the last Q-table.
    rospy.on_shutdown(env.on_shutdown)

    qInit = qlearn.QLearn(actions=range(env.nA), states=env.state_space,
                          alpha=en.Config.Q_ALPHA, gamma=en.Config.Q_GAMMA, epsilon=en.Config.Q_EPSILON)
    try:
        qInit.loadModel(base_filename + ".npy")
    except:
        print "ERROR: Q-table is not found"
        pass

    qlAgent = qlearn.QLearn(actions=range(env.nA), states=env.state_space,
                        alpha=en.Config.Q_ALPHA, gamma=en.Config.Q_GAMMA, epsilon=en.Config.Q_EPSILON, Q=qInit.Q)
    start_time = time.time()
    highest_reward = 0
    last_time_steps = numpy.ndarray(0)
    episodeRewardLog = []

    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        except:
            pass

    state, _ = env.discretize_observation(data, en.Config.DISCRETIZE_RANGE)

    E = np.zeros_like(qlAgent.Q)
    print state
    cumulated_reward = 0  # Should going forward give more reward then L/R ?
    i = 0
    while not rospy.is_shutdown():
        # Pick an action based on the current state
        action = qlAgent.chooseAction(state)
        # Execute the action and get feedback
        nextState, reward, done, info = env.step(action)
        cumulated_reward += reward
        #print "B"
        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward

        E[state, action] = 1.0

        qlAgent.learn_Q_ellgibility_trace(state, action, reward, nextState, E)

        E = E * qlAgent.gamma

        if not (done):
            state = nextState
        else:
            last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
            i = i + 1
            episodeRewardLog.append(cumulated_reward)
            state = handle_collision(env, qlAgent, base_filename, episodeRewardLog,
                                     action)  # which sends the reverse of the last action and
            cumulated_reward = 0
            # break

    rospy.spin()


if __name__ == '__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass

    # rospy.init_node('rl_agent_tb')
    # env = lidar_env.Turtlebot_Lidar_Env()

    # base_filename = 'Qlearning'

## NUM_SIMULATIONS = 5
## SAVE_FREQ = 5
## TOTAL_EPISODES = 10

# qInit = qlearn.QLearn(actions=range(env.nA), states=env.state_space,
# alpha=0.2, gamma=0.8, epsilon=0.1)

# try:
# qInit.loadModel("Qinit_" + base_filename + ".npy")
# except:
# print "ERROR: Q-table is not found"
# pass


# for sim_num in range(NUM_SIMULATIONS):
# print "-=-=-=-=-=-=-=-=-=-=-= SIMULATION " + str(sim_num + 1) + " =-=-=-=-=-=-=-=-=-=-=-"

# qlAgent = qlearn.QLearn(actions=range(env.nA), states=env.state_space,
# alpha=0.2, gamma=0.8, epsilon=0.1, Q=qInit.Q)

# initial_epsilon = qlAgent.epsilon

# epsilon_discount = 0.9986

# start_time = time.time()
# highest_reward = 0

# last_time_steps = numpy.ndarray(0)

# episodeRewardLog = []
# for x in range(TOTAL_EPISODES):
# done = False

# cumulated_reward = 0 #Should going forward give more reward then L/R ?

# state = env.reset_env()

# if qlAgent.epsilon > 0.05:
# qlAgent.epsilon *= epsilon_discount

# E = np.zeros_like(qlAgent.Q)
# for i in range(500):
## Pick an action based on the current state
# action = qlAgent.chooseAction(state)
## Execute the action and get feedback
# nextState, reward, done, info = env.step(action)
# cumulated_reward += reward

# if highest_reward < cumulated_reward:
# highest_reward = cumulated_reward


# E[state, action] = 1.0
# qlAgent.learn_Q_ellgibility_trace(state, action, reward, nextState, E)
# E = E * qlAgent.gamma

##qlAgent.learn_Q(state, action, reward, nextState)

##print "range" , i, "Done ", done
# if not(done):
# state = nextState
# else:
# last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
## TODO: handle_collision(), which sends the reverse of the last action.
# break

# episodeRewardLog.append(cumulated_reward)

# if (x + 1) % SAVE_FREQ == 0:
# print "Saving model and training log with " + base_filename + " as base filename."
# filename = "Qinit_" + str(sim_num) + base_filename
# qlAgent.saveModel(filename)
# filename = "trainingRewardLog_" + str(sim_num) + base_filename
# np.save(filename, np.asarray(episodeRewardLog))


# m, s = divmod(int(time.time() - start_time), 60)
# h, m = divmod(m, 60)
# print ("EP: "+str(x+1)+" - [alpha: "+str(round(qlAgent.alpha,2))+" - gamma: "+str(round(qlAgent.gamma,2))+" - epsilon: "+str(round(qlAgent.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))
# quit()
