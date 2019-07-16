#! /usr/bin/env python
import os
import sys
import yaml
import time
import rospy
import numpy as np

import random


import qlearn
from rl_agent import Agent
from environment import TurtlebotLIDAREnvironment as Environment

from sensor_msgs.msg import LaserScan

if __name__ == '__main__':
    # NOTE: This part should be commented to be able to run debug it in Pycharm.
#    if len(sys.argv) < 2:
#        print('run: python <script> <config>')
#        sys.exit(1)
#    script = sys.argv[0]
#    try:
#        config = yaml.load(open(sys.argv[1]))
#    except:
#        config = {}

    # NOTE: activate this part to debug the code
    config = yaml.load(open('config_real.yaml'))
    rospy.init_node('rl_agent_tb')

    experiment_name = config.setdefault('experiment_name', '')

    try:
        os.mkdir(experiment_name)
    except OSError:
        print "Experiment directory "+experiment_name+" either exists or is not a valid path. Please provide a valid path and delete previously existing directory if it exists."
        quit()

    save_lidar = config.setdefault("save_lidar", False)
    if save_lidar:
        save_lidar = experiment_name+"/LidarData"
        try:
            os.mkdir(save_lidar)
        except OSError:
            pass

    save_q = config.setdefault('save_q', False)
    if save_q:
        try:
            os.mkdir(experiment_name + "/Q" + "Episodes")
        except OSError:
            pass

    episodes = config.setdefault('episodes', 100)
    save_rewards = config.setdefault("save_rewards", True)
    if save_rewards:
        try:
            os.mkdir(experiment_name+"/RewardLogs"+str(episodes)+"Episodes")
        except OSError:
            pass
    env = Environment(save_lidar=save_lidar, **config['Environment'])

    base_filename = 'Qlearning'
    # Save the last Q-table.
    #rospy.on_shutdown(env.on_shutdown)

    agent = Agent(nA=env.A.size, nS=env.S.space_size, **config['RLAgent'])

    try:
        agent.agent.load_model(base_filename + ".npy")
    except:
        print "ERROR: Q-table is not found"
        pass

    state = env.S.state(env.A.prev_action)

    agent.reset_ellgibility_trace()
    cumulated_reward = 0
    highest_reward = 0
    last_time_steps = np.ndarray(0)
    episode_reward_log = []
    #while not rospy.is_shutdown():
    for i in range(500):
        # Pick an action based on the current state
        action_idx = agent.chooseAction(state)
        # Execute the action and get feedback
        next_state, reward, done = env.step(action_idx)

        cumulated_reward += reward
        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward

        agent.learn(state, action_idx, reward, next_state)

        if not (done):
            state = next_state
        else:
            last_time_steps = np.append(last_time_steps, [int(i + 1)])
            env.handle_collision(action_idx)
            break
            # Save the last Q matrix when there is a collision
    episode_reward_log.append(cumulated_reward)
    if save_q:
        print "Saving model and training log with " + base_filename + " as base filename."
        filename = "Qinit_" + base_filename
        agent.agent.save_model(filename)
    if save_rewards:
        filename = experiment_name + "/RewardLogs" + str(episodes) + "Episodes/" + str(1)
        np.save(filename, np.asarray(episode_reward_log))

    rospy.spin()