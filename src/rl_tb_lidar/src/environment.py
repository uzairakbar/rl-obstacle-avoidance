#! /usr/bin/env python
import yaml
import time
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty as EmptySrv
from kobuki_msgs.msg import BumperEvent
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


from utils.teleporter import Teleporter
from utils.space import ActionSpace, StateSpace

C = -10
STEP_TIME = 0.15

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



class TurtleBotRealEnvironment():
    def __init__(self, map, **kwargs):
        save_lidar = kwargs.setdefault('save_lidar', False)
        self.A = ActionSpace(**kwargs['ActionSpace'])
        self.S = StateSpace(save_lidar=save_lidar, **kwargs['StateSpace'])

        self.teleporter = Teleporter(map)

        self.is_crashed = False
        self.crash_tracker = rospy.Subscriber('mobile_base/events/bumper', BumperEvent, self.crash_callback)
        self.reset_stage = rospy.ServiceProxy('reset_positions', EmptySrv)
        return

    def crash_callback(self, data):
        if data.state == BumperEvent.PRESSED:
            self.is_crashed = True
        else:
            self.is_crashed = False

    def handle_collision(self, last_action):

        # get the last action as [Vx, theta]
        action = self.A.action(last_action, execute=False)
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0] * (-3)
        vel_cmd.angular.z = -action[1]

        # execute action
        vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=5)
        vel_pub.publish(vel_cmd)
        time.sleep(1)

        # read the new state
        state = self.S.state(last_action)
        return state


    def reward_function(self, velocities, crashed):
        if crashed:
            reward = C
        else:
            reward = velocities[0]*np.cos(velocities[1])*STEP_TIME
        return reward

    def step(self, action_idx):
        action = self.A.action(action_idx, execute=True)
        state = self.S.state(action)
        crashed = self.is_crashed
        reward = self.reward_function(action, crashed)
        return state, reward, crashed
