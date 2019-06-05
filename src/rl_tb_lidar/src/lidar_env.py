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

class Turtlebot_Lidar_Env:
    def __init__(self, nA = 7):
        self.vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=5)
        # Change this for gazebo implementation.
        rospy.wait_for_service('reset_positions')
        self.reset_stage = rospy.ServiceProxy('reset_positions', EmptySrv)
        self.state_space = range(MAX_RANGE ** (DISCRETIZE_RANGE))
        self.nS = len(self.state_space)
        self.reward_range = (-np.inf, np.inf)
        self.state_aggregation = "MIN"

        self.prev_action = np.zeros(2)
        self.nA = nA
        self.action_space = list(np.linspace(0, self.nA, 1))
        linear_velocity_list = [0.4, 0.2]
        angular_velocity_list = [np.pi/6, np.pi/12, 0., -np.pi/12, -np.pi/6]
        if self.nA == 7:
            self.action_table = linear_velocity_list + angular_velocity_list
        elif self.nA == 10:
            self.action_table = [np.array([v, w]) for v in linear_velocity_list for w in angular_velocity_list]

        # self._seed()

    def reward_function(self, action, done):
        c = -10.0
        reward = action[0]*np.cos(action[1])*STEP_TIME
        if done:
            reward = c
        return reward

    def action1(self, action_idx):
        action = self.prev_action
        if action_idx < 2:
            action[0] = self.action_table[action_idx]
        else:
            action[1] = self.action_table[action_idx]
        return action

    def action2(self, action_idx):
        action = self.action_table[action_idx]
        return action

    def discretize_observation(self, data, new_ranges):
        discrete_state = 0
        min_range = 0.3
        done = False
        ranges = data.ranges[90:270]
        if self.state_aggregation == "MIN":
            mod = len(ranges) / new_ranges
            for i in range(new_ranges):

                discrete_state = discrete_state * MAX_RANGE
                aggregator = min(ranges[mod * i : mod * (i+1)])

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

    def step(self, action_idx):
        if self.nA == 7:
            action = self.action1(action_idx)
        elif self.nA == 10:
            action = self.action2(action_idx)

        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)

        time.sleep(STEP_TIME)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.discretize_observation(data, DISCRETIZE_RANGE)

        reward = self.reward_function(action, done)
        self.prev_action = action

        return state, reward, done, {}

