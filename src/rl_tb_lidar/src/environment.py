#! /usr/bin/env python
import yaml
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty as EmptySrv
from sensor_msgs.msg import LaserScan

from utils.teleporter import Teleporter
from utils.space import ActionSpace, StateSpace

C = -10
STEP_TIME = 0.0666

class TurtlebotLIDAREnvironment():
    def __init__(self, map, **kwargs):
        save_lidar = kwargs.setdefault('save_lidar', False)
        self.A = ActionSpace(**kwargs['ActionSpace'])
        self.S = StateSpace(save_lidar=save_lidar, **kwargs['StateSpace'])

        self.teleporter = Teleporter(map)

        self.is_crashed = False
        self.crash_tracker = rospy.Subscriber('/odom', Odometry, self.crash_callback)
        self.reset_stage = rospy.ServiceProxy('reset_positions', EmptySrv)
        return

    def reward_function(self, velocities, crashed):
        if crashed:
            reward = C
        else:
            reward = velocities[0]*np.cos(velocities[1])*STEP_TIME
        return reward

    def crash_callback(self, data):
        if data.twist.twist.linear.z:
            self.is_crashed = True
        else:
            self.is_crashed = False

    def reset_env(self):
        try:
            self.teleporter.teleport_predefined()
        except (rospy.ServiceException) as e:
            print ("reset_simulation service call failed")
        state = self.S.state(self.A.prev_action)
        return state

    def step(self, action_idx):
        action = self.A.action(action_idx, execute=True)
        state = self.S.state(action)
        crashed = self.is_crashed
        reward = self.reward_function(action, crashed)
        return state, reward, crashed
