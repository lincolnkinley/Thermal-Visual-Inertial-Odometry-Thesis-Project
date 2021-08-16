#!/usr/bin/env python2

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np

PATH = "/home/lincoln/lab/data/bags_as_images/isec_front_ground/"

bridge = CvBridge()

img_num = 0

def callback(data):
    global img_num
    img = bridge.imgmsg_to_cv2(data)
    if not(img is None):
        img_name = "{:08d}.png".format(img_num)
        cv2.imwrite(PATH + img_name, img)
        img_num += 1


def main():
    rospy.init_node('qnd_video2img', anonymous=False)
    rospy.Subscriber("lepton_processed", Image, callback)
    rospy.spin()


if __name__ == "__main__":
    main()
