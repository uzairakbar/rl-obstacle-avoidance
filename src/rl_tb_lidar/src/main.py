#! /usr/bin/env python
import os
import sys
import yaml
import time
import rospy
import numpy as np

from rl_agent import Agent
from environment import TurtleBotRealEnvironment as Environment


if __name__ == '__main__':
    #if len(sys.argv) < 2:
    #    print('run: python <script> <config>')
    #    sys.exit(1)
    script = "main.py"
    try:
        config = yaml.load(open("config.yaml"))
    except:
        config = {}

    # init
    rospy.init_node('rl_agent_tb')
    env = Environment(save_lidar=False, **config['Environment'])
    agent = Agent(nA=env.A.size, nS=env.S.space_size, **config['RLAgent'])
    agent.load_model(**config["saved_model"])
    cumulated_reward = 0
    i = 0

    # read initial state
    state = env.S.state(0)

    # start interaction with environment
    while not rospy.is_shutdown():
        # Pick an action based on the current state
        action_idx = agent.chooseAction(state)

        # Execute the action and get feedback
        next_state, reward, done = env.step(action_idx)

        # Lean
        agent.learn(state, action_idx, reward, next_state)

        if not (done):
            state = next_state
        else:
            # undo last action
            state = env.handle_collision(action_idx)
            cumulated_reward = 0


    quit()
