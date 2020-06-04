#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image

def callback(msg):
    print(test)

def prediction():
    rospy.init_node('prediction', anonymous=True)
    rospy.Subscriber("/image_raw", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    prediction()
