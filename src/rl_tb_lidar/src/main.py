#! /usr/bin/env python
import os
import sys
import yaml
import time
import rospy
import numpy as np

from rl_agent import Agent
from environment import TurtlebotLIDAREnvironment as Environment


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('run: python <script> <config>')
        sys.exit(1)
    script = sys.argv[0]
    try:
        config = yaml.load(open(sys.argv[1]))
    except:
        config = {}

    rospy.init_node('rl_agent_tb')

    experiment_name = config.setdefault('experiment_name', '')
    simulations = config.setdefault('simulations', 1)
    episodes = config.setdefault('episodes', 100)
    save_q = config.setdefault('save_q', False)
    if save_q:
        try:
            os.mkdir(experiment_name+str(episodes)+"Q")
        except OSError:
            pass
    save_rewards = config.setdefault('save_rewards', True)
    if save_rewards:
        try:
            os.mkdir(experiment_name+str(episodes)+"RewardLogs")
        except OSError:
            pass
    save_lidar = config.setdefault('save_lidar', False)
    if save_lidar:
        save_lidar = experiment_name+'LidarData'
        try:
            os.mkdir(save_lidar)
        except OSError:
            pass
    save_freq = config.setdefault('save_freq', 10)

    env = Environment(save_lidar=save_lidar, **config['Environment'])
    for simulation in range(simulations):
        print "-=-=-=-=-=-=-=-=-=-=-= SIMULATION " + str(simulation + 1) + " =-=-=-=-=-=-=-=-=-=-=-"
        if env.S.space_type == 1:
            agent = Agent(nA=env.A.size, nS=env.S.reducer.levels**env.S.reducer.size, **config['RLAgent'])
        else:
            agent = Agent(nA=env.A.size, nS=(env.S.reducer.levels**env.S.reducer.size)*env.A.size, **config['RLAgent'])

        # loging stuff
        start_time = time.time()
        highest_reward = 0
        last_time_steps = np.ndarray(0)
        episode_reward_log = []
        for episode in range(episodes):
            done = False
            cumulated_reward = 0
            state = env.reset_env()
            for i in range(500):
                # Pick an action based on the current state
                action_idx = agent.chooseAction(state)
                # Execute the action and get feedback
                next_state, reward, done = env.step(action_idx)

                cumulated_reward += reward
                if highest_reward < cumulated_reward:
                    highest_reward = cumulated_reward

                agent.learn(state, action_idx, reward, next_state)

                if not(done):
                    state = next_state
                else:
                    last_time_steps = np.append(last_time_steps, [int(i + 1)])
                    break

            episode_reward_log.append(cumulated_reward)

            if (episode + 1) % save_freq == 0:
                print "Saving model/training log with " + experiment_name + " as base filename."
                if save_q:
                    filename = experiment_name+str(episodes)+"Q/"+experiment_name+"Sim"+str(simulation)+"Q"
                    agent.save_model(filename)
                if save_rewards:
                    filename = experiment_name+str(episodes)+"RewardLogs/"+experiment_name+"Sim"+str(simulation)+"RewardLogs"
                    np.save(filename, np.asarray(episode_reward_log))

            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)
            print ("EP: "+str(episode+1)+"\t Reward: "+str(cumulated_reward)+"\t Time: %d:%02d:%02d" % (h, m, s))
    quit()
