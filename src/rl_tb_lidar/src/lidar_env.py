#! /usr/bin/env python

import rospy
import time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty as EmptySrv
import numpy as np

DISCRETIZE_RANGE = 6
MAX_RANGE = 5 # max valid range is MAX_RANGE -1
STEP_TIME = 0.14  # waits 0.2 sec after the action

ACTION_FORWARD = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2


class Turtlebot_Lidar_Env:
    def __init__(self):
        self.vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=5)
        # Change this for gazebo implementation.
        rospy.wait_for_service('reset_positions')
        self.reset_stage = rospy.ServiceProxy('reset_positions', EmptySrv)
        self.action_space = [0, 1, 2]  # F,L,R
        self.state_space = range(MAX_RANGE ** (DISCRETIZE_RANGE))
        self.nA = len (self.action_space)
        self.nS = len(self.state_space)
        self.reward_range = (-np.inf, np.inf)
        self.state_aggregation = "MIN"

        # self._seed()

    def discretize_observation(self, data, new_ranges):
        discrete_state = 0
        min_range = 0.3
        done = False

        if self.state_aggregation == "MIN":
            mod = len(data.ranges) / new_ranges
            for i in range(new_ranges):

                discrete_state = discrete_state * MAX_RANGE
                aggregator = min(data.ranges[mod * i : mod * (i+1)])

                if aggregator > 2.5:
                    aggregator = 4
                elif aggregator > 1.5:
                    aggregator = 3
                elif aggregator > 1:
                    aggregator = 2
                elif aggregator > 0.5:
                    aggregator = 1
                else:
                    aggregator = 0

                if np.isnan(aggregator):
                    discrete_state = discrete_state
                else:
                    discrete_state = discrete_state + int(aggregator)

            if min_range > min(data.ranges):
                done = True

            return discrete_state, done


        mod = len(data.ranges) / new_ranges
        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                discrete_state = discrete_state * MAX_RANGE

                if data.ranges[i] == float('Inf') or np.isinf(data.ranges[i]):
                    discrete_state = discrete_state +  6
                elif np.isnan(data.ranges[i]):
                    discrete_state = discrete_state
                else:
                    discrete_state = discrete_state  + int (data.ranges[i])
            if (min_range > data.ranges[i] > 0):
                done = True
        return discrete_state, done


    def reset_env(self):
        rospy.wait_for_service('reset_positions')
        try:
            # reset_proxy.call()
            self.reset_stage()
        except (rospy.ServiceException) as e:
            print ("reset_simulation service call failed")

        # read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        state, _ = self.discretize_observation(data, DISCRETIZE_RANGE)

        return state

    def step(self, action):
        if action == 0:  # FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1:  # LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = 0.3
            self.vel_pub.publish(vel_cmd)
        elif action == 2:  # RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = -0.3
            self.vel_pub.publish(vel_cmd)

        time.sleep(STEP_TIME)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.discretize_observation(data, DISCRETIZE_RANGE)

        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 1
        else:
            reward = -200

        return state, reward, done, {}

