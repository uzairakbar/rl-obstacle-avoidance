#! /usr/bin/env python
import rospy
import numpy
from sensor_msgs.msg import LaserScan

# Method 1: receives e.g. 4 measurements from /scan and replace "inf" in the the last measurement with
# the measured data from the previous measurements to define the state. It waits all the MEASUREMENTS to publish the data.
# Method 2: consecutively publish the filtered data for state representation.
METHOD = 2

# Number of observations required the state of the turtlebot.
# Note: Lidar frequency is 5.5Hz
MEASUREMENTS = 4

class Lidar_Filter:
    def __init__(self):
        self.laser_filtered_pub = rospy.Publisher('/scan_filtered', LaserScan, queue_size=5)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.process_laser)
        self.scan_values = []   # is used to store the scanned lidar values.
        inf = float("inf")
        self.scan_values_2 = [[inf for col in range(360)] for row in range(MEASUREMENTS)]
        self.counter = 0        # is used to count the number of measurements.
        self.init = False       # is used for the second method. We wait if we do not have any measurements at the beginning.
        self.init_counter = 0   # initilization counter to at first receive some measurements results.
        rospy.spin()

    def process_laser(self, data):
        """
        Callback function to process the received scan values.
        :param data:
        :return:
        """
        if METHOD == 1:
            if self.counter < MEASUREMENTS:
                self.scan_values.append(data.ranges)
                self.counter += 1
            else:
                self.filter_values()
        else:
            counter = data.header.seq % MEASUREMENTS
            self.scan_values_2[counter] = data.ranges
            self.filter_values_2(counter)

            # Apply the second method to publish the filtered data.

    def filter_values_2(self, counter):
        """
        This filter is used for method 2
        :param counter:
        :return:
        """
        inf = float("inf")
        last_mea = list(self.scan_values_2[counter])
        for i in range(len(last_mea)):
            if last_mea[i] == inf:
                for k in range(1, MEASUREMENTS):
                    if self.scan_values_2[counter-k] != inf:
                        last_mea[i] = self.scan_values_2[k][i]
                        break
        scan = LaserScan()
        scan.ranges = last_mea
        self.laser_filtered_pub.publish(scan)

    def filter_values_1(self):
        """
        Filters laser values for the first method.
        Replace the inf numbers with a value from the other measurements
        starting from the last measurement to previous ones.
        :return:
        """
        inf = float("inf")
        last_mea = list(self.scan_values[-1])
        for i in range(len(last_mea)):
            if last_mea[i] == inf:
                for k in reversed(xrange(MEASUREMENTS-1)):
                    if self.scan_values[k][i] != inf:
                        last_mea[i] = self.scan_values[k][i]
                        break
        scan = LaserScan()
        scan.ranges = last_mea
        self.laser_filtered_pub.publish(scan)
        self.counter = 0
        self.scan_values = []

def run():
    rospy.init_node('lidar_filter')
    filter = Lidar_Filter()

if __name__ == '__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass

