#!/usr/bin/env python

import unittest
import rospy
import time
import Queue as queue
from sensor_msgs.msg import LaserScan

PKG = "stage_ros"


class RangerIntensityTests(unittest.TestCase):
    def setUp(self):
        """Called before every test. Set up a LaserScan subscriber
        """
        rospy.init_node('ranger_intensity_test', anonymous=True)
        self._scan_q = queue.Queue()
        self._scan_topic = rospy.get_param("scan_topic", "base_scan")
        self._subscriber = rospy.Subscriber(self._scan_topic,
                                            LaserScan, self._scan_callback)

    def tearDown(self):
        """Called after every test. Cancel the scan subscription
        """
        self._subscriber.unregister()
        self._scan_q = None
        self._subscriber = None

    def _scan_callback(self, scan_msg):
        """Called every time a scan is received
        """
        self._scan_q.put(scan_msg)

    def _wait_for_scan(self, timeout=1.0):
        """Wait for a laser scan to be received
        """
        # Use wall clock time for timeout
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                return self._scan_q.get(True, 0.1)
            except queue.Empty:
                pass
        return None

    def test_intensity_greater_than_256(self):
        """Make sure stage_ros returns intensity values higher than 256
        """
        scan = self._wait_for_scan()
        self.assertIsNotNone(scan)
        self.assertGreater(max(scan.intensities), 256.9)


if __name__ == '__main__':
    import rosunit
    rosunit.unitrun(PKG, 'test_intensity', RangerIntensityTests)
