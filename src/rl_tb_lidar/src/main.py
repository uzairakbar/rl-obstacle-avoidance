#! /usr/bin/env python
import os
import sys
import yaml
import time
import rospy
import numpy as np

# from rl_agent import Agent
from agent import Agent
from environment import TurtlebotLIDAREnvironment as Environment


if __name__ == '__main__':
    # NOTE: This part should be commented to be able to run debug it in Pycharm.
    if len(sys.argv) < 2:
        print('run: python <script> <config>')
        sys.exit(1)
    script = sys.argv[0]
    try:
        config = yaml.load(open(sys.argv[1]))
    except:
        config = {}

    # NOTE: activate this part to debug the code
    #config = yaml.load(open('configs/config_domain_randomization.yaml'))
    rospy.init_node('rl_agent_tb')

    experiment_name = config.setdefault('experiment_name', '')
    simulations = config.setdefault('simulations', 1)
    episodes = config.setdefault('episodes', 100)

    try:
        os.mkdir(experiment_name)
        with open(experiment_name+'/config.yaml', 'w') as config_file:
            yaml.dump(config, config_file, default_flow_style=False)
    except OSError:
        print "Exeriment directory "+experiment_name+" either exists or is not a valid path. Please provide a valid path and delete previously existing directory if it exists."
        quit()

    save_q = config.setdefault('save_q', False)
    if save_q:
        try:
            os.mkdir(experiment_name+"/Q"+str(episodes)+"Episodes")
        except OSError:
            pass

    save_rewards = config.setdefault("save_rewards", True)
    if save_rewards:
        try:
            os.mkdir(experiment_name+"/RewardLogs"+str(episodes)+"Episodes")
        except OSError:
            pass

    save_lidar = config.setdefault("save_lidar", False)
    if save_lidar:
        save_lidar = experiment_name+"/LidarData"
        try:
            os.mkdir(save_lidar)
        except OSError:
            pass
    save_freq = config.setdefault("save_freq", 100)

    env = Environment(save_lidar=save_lidar, **config['Environment'])
    for simulation in range(simulations):
        print "-=-=-=-=-=-=-=-=-=-=-= SIMULATION " + str(simulation + 1) + " =-=-=-=-=-=-=-=-=-=-=-"
        if config['RLAgent']['lvfa']:
            agent = Agent(nA=env.A.size, nS=env.S.size, episodes = episodes, **config['RLAgent'])
        else:
            agent = Agent(nA=env.A.size, nS=env.S.space_size, episodes = episodes, **config['RLAgent'])

        # loging stuff
        start_time = time.time()
        highest_reward = 0
        last_time_steps = np.ndarray(0)
        episode_reward_log = []
        for episode in range(episodes):
            done = False
            cumulated_reward = 0
            state = env.reset_env()
            agent.reset_eligibility()
            
            if config['RLAgent']['algorithm'] == 'sarsa':
                action_idx = agent.action(state, episode)
            elif config['RLAgent']['algorithm'] == 'qlearning':
                next_action_idx = None
            for i in range(500):
                if config['RLAgent']['algorithm'] == 'sarsa':
                    next_state, reward, done = env.step(action_idx)
                    next_action_idx = agent.action(next_state, episode)
                elif config['RLAgent']['algorithm'] == 'qlearning':
                    action_idx = agent.action(state, episode)
                    next_state, reward, done = env.step(action_idx)
                
                agent.learn(state,
                            action_idx,
                            reward,
                            next_state,
                            next_action_idx)

                cumulated_reward += reward
                if highest_reward < cumulated_reward:
                    highest_reward = cumulated_reward

                if not(done):
                    action_idx = next_action_idx
                    state = next_state
                else:
                    last_time_steps = np.append(last_time_steps, [int(i + 1)])
                    break

            episode_reward_log.append(cumulated_reward)

            if (episode + 1) % save_freq == 0:
                print "Saving model/training log with " + experiment_name + " as base filename."
                if save_q:
                    filename = experiment_name+"/Q"+str(episodes)+"Episodes/"+experiment_name+"_sim"+str(simulation)
                    agent.save_model(filename)
                if save_rewards:
                    filename = experiment_name+"/RewardLogs"+str(episodes)+"Episodes/"+experiment_name+"_sim"+str(simulation)
                    np.save(filename, np.asarray(episode_reward_log))

            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)
            print ("EP: "+str(episode+1)+"\t Reward: "+str(cumulated_reward)+"\t Time: %d:%02d:%02d" % (h, m, s))
    quit()
