#! /usr/bin/env python
import rospy
import time
import random
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Int8
from std_srvs.srv import Empty as EmptySrv
import numpy as np

from kobuki_msgs.msg import BumperEvent



is_crashed = False


class Turtlebot_Lidar_Env:
    def __init__(self, nA=10):
        # TODO check how to send a command to move robot.
        self.vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=5)
        # Change this for gazebo implementation.
        # rospy.wait_for_service('reset_positions')

        # self.reset_stage = rospy.ServiceProxy('reset_positions', EmptySrv)
        # self.teleporter = rospy.Publisher('/cmd_pose', Pose, queue_size=10)
        # TODO change this part to track /bumper topic in order to understand the crash
        # Change also the callback function such that if the turtlebot hit an obstacle it moves back e.g. 0.5m and continue from that state.
        #self.crash_tracker = rospy.Subscriber('/odom', Odometry, self.crash_callback)

        self.bumber_sub = rospy.Subscriber('mobile_base/events/bumper', BumperEvent, self.process_bump)
        self.state_space = range(Config.MAX_RANGE ** (Config.DISCRETIZE_RANGE))
        self.nS = len(self.state_space)
        self.reward_range = (-np.inf, np.inf)
        self.state_aggregation = "MIN"

        # self.map = map
        self.prev_action = np.zeros(2)
        self.nA = nA
        self.action_space = list(np.linspace(0, self.nA, 1))
        linear_velocity_list = [0.4, 0.2]
        angular_velocity_list = [np.pi / 6, np.pi / 12, 0., -np.pi / 12, -np.pi / 6]
        if self.nA == 7:
            self.action_table = linear_velocity_list + angular_velocity_list
        elif self.nA == 10:
            self.action_table = [np.array([v, w]) for v in linear_velocity_list for w in angular_velocity_list]

        # self._seed()

    def process_bump(self, data):
        print ("Bump")
        global is_crashed
        if (data.state == BumperEvent.PRESSED):
            is_crashed = True
        else:
            is_crashed = False
        rospy.loginfo("Bumper Event")
        rospy.loginfo(data.bumper)

    def crash_callback(self, data):
        global is_crashed
        # Add the line below to read current velocity value.
        # print data.twist.twist.linear.x
        if data.twist.twist.angular.z:
            is_crashed = True
        else:
            is_crashed = False


    def reward_function(self, action, done):
        c = -10.0
        reward = action[0] * np.cos(action[1]) * Config.STEP_TIME
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
        a = data.ranges[0:60]
        b = data.ranges[300:360]
        data.ranges = np.concatenate((b, a), axis=None)
        if self.state_aggregation == "MIN":
            mod = len(data.ranges) / new_ranges
            for i in range(new_ranges):

                discrete_state = discrete_state * Config.MAX_RANGE
                aggregator = min(data.ranges[mod * i: mod * (i + 1)])

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

            # if min_range > min(data.ranges):
            # done = True
            if is_crashed:
                done = True
                # self.teleport_random()

            return discrete_state, done

        mod = len(data.ranges) / new_ranges
        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                discrete_state = discrete_state * Config.MAX_RANGE

                if data.ranges[i] == float('Inf') or np.isinf(data.ranges[i]):
                    discrete_state = discrete_state + 6
                elif np.isnan(data.ranges[i]):
                    discrete_state = discrete_state
                else:
                    discrete_state = discrete_state + int(data.ranges[i])
            if (min_range > data.ranges[i] > 0):
                done = True
        return discrete_state, done


    def discretize_state(self, state):
        #data, v, w = state
        data = state
        discrete_observation, done = self.discretize_observation(data, Config.DISCRETIZE_RANGE)
        v_list = self.linear_velocity_list
        w_list = self.angular_velocity_list
        #try:
            #v_idx = v_list.index(v)
        #except:
            #v_idx = 0
        #w_idx = w_list.index(w)
        #discrete_state = discrete_observation + v_idx*(MAX_RANGE**DISCRETIZE_RANGE)
        #discrete_state = discrete_state + w_idx*(len(v_list)*MAX_RANGE**DISCRETIZE_RANGE)
        discrete_state = discrete_observation
        return discrete_state, done

    def reset_env(self):
        # rospy.wait_for_service('reset_positions')
        try:
            # reset_proxy.call()
            # self.reset_stage()
            # self.teleport_random()
            self.teleport_predefined()
        except (rospy.ServiceException) as e:
            print ("reset_simulation service call failed")

        # read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        state, _ = self.discretize_observation(data, Config.DISCRETIZE_RANGE)

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

        time.sleep(Config.STEP_TIME)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                # np.save(self.filename+str(self.filecounter), np.asarray(data.ranges, dtype=np.float16))
                # self.filecounter += 1
            except:
                pass

        # state, done = self.discretize_observation(data, DISCRETIZE_RANGE)
        state = data
        self.curr_state = np.append(np.asarray(data.ranges), action)
        discrete_state, done = self.discretize_state(state)

        reward = self.reward_function(action, done)
        self.prev_action = action

        return discrete_state, reward, done, {}

    def teleport_random(self):
        """
        Teleport the robot to a new random position on map
        """
        x_min = 0  # bounds of the map
        x_max = 10
        y_min = 0
        y_max = 10

        # Randomly generate a pose
        cmd_pose = Pose()
        cmd_pose.position.x = random.uniform(x_min, x_max)
        cmd_pose.position.y = random.uniform(y_min, y_max)

        cmd_pose.orientation.z = random.uniform(-7, 7)  # janky way of getting most of the angles from a quaternarion
        cmd_pose.orientation.w = random.uniform(-1, 1)
        # cmd_pose.orientation.w = 1

        # ... and publish it as the new pose of the robot
        time.sleep(0.3)
        self.teleporter.publish(cmd_pose)
        time.sleep(0.3)  # wait (in real time) before and after jumping to avoid segfaults

    def teleport_predefined(self):
        r = random.randint(1, 5)
        cmd_pose = Pose()
        cmd_pose.orientation.z = random.uniform(-7, 7)
        cmd_pose.orientation.w = random.uniform(-1, 1)
        if self.map == "map1":
            if r == 1:
                cmd_pose.position.x = 1.0
                cmd_pose.position.y = 2.0
            elif r == 2:
                cmd_pose.position.x = 5.0
                cmd_pose.position.y = 5.0
            elif r == 3:
                cmd_pose.position.x = 7.0
                cmd_pose.position.y = 8.0
            elif r == 4:
                cmd_pose.position.x = 3.0
                cmd_pose.position.y = 1.0
            else:
                cmd_pose.position.x = 6.0
                cmd_pose.position.y = 2.0
        elif self.map == "map2":
            if r == 1:
                cmd_pose.position.x = 2.0
                cmd_pose.position.y = 2.0
            elif r == 2:
                cmd_pose.position.x = 7.0
                cmd_pose.position.y = 8.0
            elif r == 3:
                cmd_pose.position.x = 2.0
                cmd_pose.position.y = 5.0
            elif r == 4:
                cmd_pose.position.x = 8.0
                cmd_pose.position.y = 3.0
            else:
                cmd_pose.position.x = 1.0
                cmd_pose.position.y = 7.0
        elif self.map == "map3":
            if r == 1:
                cmd_pose.position.x = 2.0
                cmd_pose.position.y = 2.0
            elif r == 2:
                cmd_pose.position.x = 5.0
                cmd_pose.position.y = 5.0
            elif r == 3:
                cmd_pose.position.x = 1.0
                cmd_pose.position.y = 6.0
            elif r == 4:
                cmd_pose.position.x = 5.0
                cmd_pose.position.y = 8.0
            else:
                cmd_pose.position.x = 8.0
                cmd_pose.position.y = 4.0
        else:
            print "ERROR: Map is not defined"

        # ... and publish it as the new pose of the robot
        time.sleep(0.3)
        self.teleporter.publish(cmd_pose)
        time.sleep(0.3)  # wait (in real time) before and after jumping to avoid segfaults

    def on_shutdown(self):
        # rospy.loginfo("[%s] Shutting down." %(self.node_name))
        rospy.loginfo("Shutting down....")
        self.stop_moving()
        # rospy.loginfo("Stopped %s's velocity." %(self.veh_name))

    def stop_moving(self):
        twist = Twist()
        self.vel_pub.publish(twist)

    def handle_collision(self, last_action):
        # Apply the reverse action to move the bumper back.
        if self.nA == 7:
            action = self.action1(last_action)
        elif self.nA == 10:
            action = self.action2(last_action)

        vel_cmd = Twist()
        vel_cmd.linear.x = action[0] * (-3)
        vel_cmd.angular.z = -action[1]
        self.vel_pub.publish(vel_cmd)

        time.sleep(Config.STEP_TIME)

        # read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        state, _ = self.discretize_observation(data, Config.DISCRETIZE_RANGE)

        return state


class Config:
    DISCRETIZE_RANGE = 6
    MAX_RANGE = 5  # max valid range is MAX_RANGE -1
    STEP_TIME = 0.14  # waits 0.2 sec after the action
    Q_ALPHA = 0.01
    Q_GAMMA = 0.99
    Q_EPSILON = 0.1
