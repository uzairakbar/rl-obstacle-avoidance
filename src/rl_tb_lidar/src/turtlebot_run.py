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
    """
    This function is called if the turtlebot hit some obstacles. The latest Q-matrix and reward function will be stored.
    Also, turtlebot moves back from the obstacle.
    :param env:
    :param agent:
    :param base_filename:
    :param episodeRewardLog:
    :param last_action:         Last action is necessary so that turtlebot applies the reverse of the last action to get rid of the obstacle.
                                NOTE: Simple, move back function would also work fine to move away from the obstacle.
    :return:
    """
    print "Saving model and training log with " + base_filename + " as base filename."
    filename = "Qinit_" + base_filename
    agent.saveModel(filename)
    filename = "trainingRewardLog_" + base_filename
    np.save(filename, np.asarray(episodeRewardLog))
    # apple the reserve action to move the turtlebot back.
    return env.handle_collision(last_action)


def run_real():
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
            data = rospy.wait_for_message('/scan_filtered', LaserScan, timeout=5)
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
        if en.Config.RUN_REAL_TURTLEBOT:
            run_real()
    except rospy.ROSInterruptException:
        pass