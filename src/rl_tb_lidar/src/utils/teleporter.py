import random
import time
import rospy
from geometry_msgs.msg import Pose

class Teleporter(object):
    def __init__(self):
        self.teleporter = rospy.Publisher('/cmd_pose', Pose, queue_size=10)

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

        cmd_pose.orientation.z = random.uniform(-7,7)   # janky way of getting most of the angles from a quaternarion
        cmd_pose.orientation.w = random.uniform(-1, 1)

        # ... and publish it as the new pose of the robot
        time.sleep(0.3)
        self.teleporter.publish(cmd_pose)
        time.sleep(0.3)   # wait (in real time) before and after jumping to avoid segfaults

    def teleport_predefined(self, map, offset_x=0, offset_y=0):
        r = random.randint(1, 5)
        cmd_pose = Pose()
        cmd_pose.orientation.z = random.uniform(-7, 7)
        cmd_pose.orientation.w = random.uniform(-1, 1)
        cmd_pose.position.x = offset_x
        cmd_pose.position.y = offset_y
        if map == "map1":
            if r == 1:
                cmd_pose.position.x += 4.0
                cmd_pose.position.y += 2.5
            elif r == 2:
                cmd_pose.position.x += 5.0
                cmd_pose.position.y += 6.5
            elif r ==3:
                cmd_pose.position.x += 7.5
                cmd_pose.position.y += 2.4
            elif r==4:
                cmd_pose.position.x += 4.0
                cmd_pose.position.y += 3.5
            else:
                cmd_pose.position.x += 4.0
                cmd_pose.position.y += 3.5
        elif map == "map2":
            if r == 1:
                cmd_pose.position.x += 2.0
                cmd_pose.position.y += 2.0
            elif r == 2:
                cmd_pose.position.x += 7.0
                cmd_pose.position.y += 8.0
            elif r ==3:
                cmd_pose.position.x += 2.0
                cmd_pose.position.y += 5.0
            elif r==4:
                cmd_pose.position.x += 8.0
                cmd_pose.position.y += 3.0
            else:
                cmd_pose.position.x += 1.0
                cmd_pose.position.y += 7.0
        elif map == "map3":
            if r == 1:
                cmd_pose.position.x += 2.0
                cmd_pose.position.y += 2.0
            elif r == 2:
                cmd_pose.position.x += 5.0
                cmd_pose.position.y += 5.0
            elif r ==3:
                cmd_pose.position.x += 1.0
                cmd_pose.position.y += 6.0
            elif r==4:
                cmd_pose.position.x += 5.0
                cmd_pose.position.y += 8.0
            else:
                cmd_pose.position.x += 8.0
                cmd_pose.position.y += 4.0
        elif map == "map4":
            if r == 1:
                cmd_pose.position.x += 4.30
                cmd_pose.position.y += 3.75
            elif r == 2:
                cmd_pose.position.x += 4.0
                cmd_pose.position.y += 2.0
            elif r ==3:
                cmd_pose.position.x += 6.0
                cmd_pose.position.y += 6.0
            elif r==4:
                cmd_pose.position.x += 4.0
                cmd_pose.position.y += 4.0
            else:
                cmd_pose.position.x += 1.0
                cmd_pose.position.y += 3.0
        elif map == "map5":
            if r == 1:
                cmd_pose.position.x += 4.0
                cmd_pose.position.y += 4.0
            elif r == 2:
                cmd_pose.position.x += 1.0
                cmd_pose.position.y += 1.0
            elif r ==3:
                cmd_pose.position.x += 6.0
                cmd_pose.position.y += 4.0
            elif r==4:
                cmd_pose.position.x += 2.5
                cmd_pose.position.y += 2.5
            else:
                cmd_pose.position.x += 3.0
                cmd_pose.position.y += 5.0
        elif map == "map6":
            if r == 1:
                cmd_pose.position.x += 1.0
                cmd_pose.position.y += 2.0
            elif r == 2:
                cmd_pose.position.x += 4.30
                cmd_pose.position.y += 3.75
            elif r ==3:
                cmd_pose.position.x += 3.0
                cmd_pose.position.y += 5.0
            elif r==4:
                cmd_pose.position.x += 6.0
                cmd_pose.position.y += 3.0
            else:
                cmd_pose.position.x += 4.0
                cmd_pose.position.y += 1.0
        else:
            print "ERROR: Map is not defined"

        # ... and publish it as the new pose of the robot
        time.sleep(0.3)
        self.teleporter.publish(cmd_pose)
        time.sleep(0.3)   # wait (in real time) before and after jumping to avoid segfaults

    def teleport_domain_randomization(self):
        r = random.randint(1, 6)
        if r == 1:
            self.teleport_predefined(map="map1", offset_x=0, offset_y=10)
        elif r == 2:
            self.teleport_predefined(map="map2", offset_x=10, offset_y=10)
        elif r == 3:
            self.teleport_predefined(map="map3", offset_x=20, offset_y=10)
        elif r == 4:
            self.teleport_predefined(map="map4", offset_x=0, offset_y=0)
        elif r == 5:
            self.teleport_predefined(map="map5", offset_x=10, offset_y=0)
        elif r == 6:
            self.teleport_predefined(map="map6", offset_x=20, offset_y=0)
        else:
            print "This is not possible !!!"
