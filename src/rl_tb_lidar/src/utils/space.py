import numpy as np


class Space(object):
    def __init__(self, space):
        self.space = space
        if self.space is None:
            self.size = np.infty
        else:
            self.size = len(self.space)

    def sample(self):
        sample = np.random.choice(self.space)
        return sample


import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class ActionSpace(Space):
    def __init__(self,
                 space_type=1,
                 read_sensor=False,
                 **kwargs):
        self.space_type = space_type
        self.linear_velocity_list = [0.4, 0.2]
        self.angular_velocity_list = [np.pi/6, np.pi/12, 0., -np.pi/12, -np.pi/6]

        self.prev_action = np.asarray([0.2, 0.])
        if read_sensor:
            self.prev_action_tracker = rospy.Subscriber('/odom', Odometry, self.velocity_tracker)
        if self.space_type == 1:
            space = [np.asarray([v, w]) for v in self.linear_velocity_list for w in self.angular_velocity_list]
        elif self.space_type == 2:
            space = self.linear_velocity_list + self.angular_velocity_list
        else:
            raise ValueError("space can only be either 1 or 2, but got "+str(space)+".")
        self.vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=5)
        return super(ActionSpace, self).__init__(space)

    def action(self, action_idx, execute=False):
        if self.space_type == 1:
            action = self.space[action_idx]
        elif self.space_type == 2:
            action = self.prev_action
            if action_idx < 2:
                action[0] = self.space[action_idx]
            else:
                action[1] = self.space[action_idx]
        if execute:
            self.execute(action)
        return action

    def action_space(self, action_idx):
        if self.space_type == 1:
            return self.space
        elif self.space_type == 2:
            velocities = self.prev_action
            np.all(sample != self.prev_action)
            if action_idx < 2:
                action[0] = self.space[action_idx]
            else:
                action[1] = self.space[action_idx]
        if execute:
            self.execute(action)
        return action

    def sample(self, execute=False):
        sample_idx = np.random.randint(self.size)
        sample = self.action(sample_idx, False)
        if self.space_type == 2:
            while np.all(sample != self.prev_action):
                sample_idx = np.random.randint(self.size)
                sample = self.action(sample_idx, False)
        if execute:
            self.execute(sample)
        return sample

    def execute(self, action):
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.prev_action = action

    def discretize_velocity(self, action):
        v_idx = np.argmin(np.square(np.asarray(self.linear_velocity_list) - action[0]))
        w_idx = np.argmin(np.square(np.asarray(self.angular_velocity_list) - action[1]))
        v = self.linear_velocity_list[v_idx]
        w = self.angular_velocity_list[w_idx]
        return np.asarray([v, w])

    def velocity_tracker(self, data):
        action = self.prev_action.copy()
        action[0] = data.twist.twist.linear.x
        action[1] = data.twist.twist.angular.z
        self.prev_action = self.discretize_velocity(action)


import os
from sensor_msgs.msg import LaserScan
from utils.sensormodel.sensormodel import SampleLIDAR
from discretizer import Discretizer
from features import Features


class StateSpace(Space):
    def __init__(self,
                 space_type = 1,
                 reducer = 'discretize',
                 reducer_type = None,
                 sensor_model = 0,
                 lidar_filter = False,
                 **kwargs):
        super(StateSpace, self).__init__(None)

        if lidar_filter:
            self.lidar_filter = True
        else:
            self.lidar_filter = False

        self.space_type = space_type
        if sensor_model == 0:
            theta = [1.0, 0.0, 0.0, 0.0, 10**-8, 1.0]
        elif sensor_model == 1:
            theta = [0.5247179,  0.03035531, 0.34746964, 0.09745715, 0.19708422, 0.63124529]
        elif sensor_model == 2:
            theta = [0.50858697, 0.01769387, 0.34746964, 0.12624952, 0.00421572, 0.72616816]
        else:
            theta = sensor_model
        self.sensor_model = SampleLIDAR(theta=theta)

        if self.space_type == 1:
            pass
        elif self.space_type == 2:
            self.linear_velocity_list = [0.4, 0.2]
            self.angular_velocity_list = [np.pi/6, np.pi/12, 0., -np.pi/12, -np.pi/6]
            self.action_space = np.asarray([np.asarray([v, w]) for v in self.linear_velocity_list for w in self.angular_velocity_list])
        else:
            raise ValueError("space can only be either 1 or 2, but got "+str(space)+".")

        if reducer == 'discretize':
            self.reducer = Discretizer(**kwargs['Discretizer'])
        elif reducer == 'features':
            self.reducer = Features(**kwargs['Features'])
        else:
            raise ValueError("reducer can only be either discretize or features, but got "+reducer+" instead.")

        # check this. change to reducer.levels**reducer.size or something
        if self.space_type == 1:
            self.size = self.reducer.size
            if reducer == 'discretize':
                self.space_size = self.reducer.levels**self.reducer.size
        else:
            if reducer == 'discretize':
                self.size = self.reducer.size + 2
                self.space_size = (self.reducer.levels**self.reducer.size)*10
            else:
                self.size = self.reducer.size + 10
        self.save_lidar = kwargs.setdefault('save_lidar', False)
        if self.save_lidar:
            self.save_iterator = 0
            try:
                os.mkdir(self.save_lidar)
            except OSError:
                pass
        self.save_freq = kwargs.setdefault('save_freq', 10)

    def enumerate_state(self, enumerated_signal, velocities):
    	v, w = velocities
    	v_list = self.linear_velocity_list
    	w_list = self.angular_velocity_list

        v_idx = v_list.index(v)
    	w_idx = w_list.index(w)

        code_size = self.reducer.size
        levels = self.reducer.levels

        enumerated_state = enumerated_signal + v_idx*(levels**code_size)
    	enumerated_state = enumerated_state + w_idx*(len(v_list)*levels**code_size)

        return enumerated_state

    def state(self, velocities=None):
        signal = None
        while signal is None:
            try:
                if self.lidar_filter:
                    signal = rospy.wait_for_message('/scan_filtered', LaserScan, timeout=5)
                else:
                    signal = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                continue
        signal = np.asarray(signal.ranges)

        if self.save_lidar:
            if self.save_iterator%self.save_freq == 0:
                with open(self.save_lidar+"/lidarData.csv", 'ab') as f:
                    np.savetxt(f, signal, delimiter=",")
            self.save_iterator += 1

        noisy_signal = self.sensor_model.sample(signal)
        reduced_signal = self.reducer(noisy_signal)
        if self.space_type == 1:
            state = reduced_signal
        else:
            if isinstance(self.reducer, Discretizer):
                state = self.enumerate_state(reduced_signal, velocities)
            elif isinstance(self.reducer, Features):
                encoded_velocities = (self.action_space == np.asarray(velocities)).all(axis=1).astype(float)
                state = np.concatenate((reduced_signal, encoded_velocities))
        return state

    def sample(self, velocities=None):
        sample = self.reducer.sample()
        if self.space_type == 2:
            if isinstance(self.reducer, Discretizer):
                sample = self.enumerate_state(sample, velocities)
            elif isinstance(self.reducer, Features):
                encoded_velocities = (self.action_space == np.asarray(velocities)).all(axis=1).astype(float) # REMEMBER TO PORT THIS TO THE REAL CODE!!!!!!!!!!
                sample = np.concatenate((sample, encoded_velocities))
        return sample
