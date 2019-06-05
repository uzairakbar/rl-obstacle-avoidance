#! /usr/bin/env python
import rospy
import numpy
import random
import time

import numpy as np
import qlearn
import lidar_env

if __name__ == '__main__':
    rospy.init_node('rl_agent_tb')
    env = lidar_env.Turtlebot_Lidar_Env()

    base_filename = 'Qlearning'

    NUM_SIMULATIONS = 5
    SAVE_FREQ = 5
    TOTAL_EPISODES = 10

    qInit = qlearn.QLearn(actions=range(env.nA), states=env.state_space,
                    alpha=0.2, gamma=0.8, epsilon=0.1)

    try:
        qInit.loadModel("Qinit_" + base_filename + ".npy")
    except:
        pass

    for sim_num in range(NUM_SIMULATIONS):
        print "-=-=-=-=-=-=-=-=-=-=-= SIMULATION " + str(sim_num + 1) + " =-=-=-=-=-=-=-=-=-=-=-"

        qlAgent = qlearn.QLearn(actions=range(env.nA), states=env.state_space,
                        alpha=0.2, gamma=0.8, epsilon=0.1, Q=qInit.Q)
        
        initial_epsilon = qlAgent.epsilon

        epsilon_discount = 0.9986

        start_time = time.time()
        highest_reward = 0
        
        last_time_steps = numpy.ndarray(0)

        episodeRewardLog = []
        for x in range(TOTAL_EPISODES):
            done = False
            
            cumulated_reward = 0 #Should going forward give more reward then L/R ? 
            
            state = env.reset_env()
            
            if qlAgent.epsilon > 0.05:
                qlAgent.epsilon *= epsilon_discount

            E = np.zeros_like(qlAgent.Q)
            for i in range(500):
                # Pick an action based on the current state
                action = qlAgent.chooseAction(state)
                # Execute the action and get feedback
                nextState, reward, done, info = env.step(action)
                cumulated_reward += reward
                
                if highest_reward < cumulated_reward:
                    highest_reward = cumulated_reward


                E[state, action] = 1.0
                qlAgent.learn_Q_ellgibility_trace(state, action, reward, nextState, E)
                E = E * qlAgent.gamma

                #qlAgent.learn_Q(state, action, reward, nextState)

                #print "range" , i, "Done ", done
                if not(done):
                    state = nextState
                else:
                    last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                    break
            
            episodeRewardLog.append(cumulated_reward)

            if (x + 1) % SAVE_FREQ == 0:
                print "Saving model and training log with " + base_filename + " as base filename."
                filename = "Qinit_" + str(sim_num) + base_filename
                qlAgent.saveModel(filename)
                filename = "trainingRewardLog_" + str(sim_num) + base_filename
                np.save(filename, np.asarray(episodeRewardLog))


            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)
            print ("EP: "+str(x+1)+" - [alpha: "+str(round(qlAgent.alpha,2))+" - gamma: "+str(round(qlAgent.gamma,2))+" - epsilon: "+str(round(qlAgent.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))
    quit()

