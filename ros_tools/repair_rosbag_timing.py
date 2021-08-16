#!/usr/bin/env python2

import rospy
from cv_bridge import CvBridge
import rosbag
from sensor_msgs.msg import Image
from std_msgs.msg import Time
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Fix a rosbag timing.")
parser.add_argument("bagfile", help="Name of the bagfile used as input")
args = parser.parse_args()

    
   
def main():
    with rosbag.Bag(args.bagfile[:-4]+"_fixed.bag", 'w') as outbag:
        for topic, msg, t in rosbag.Bag(args.bagfile).read_messages():
            if topic == "/tf" and msg.transforms:
                outbag.write(topic, msg, msg.transforms[0].header.stamp)
            else:
                outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)
        
	
if __name__ == "__main__":
    main()
        
