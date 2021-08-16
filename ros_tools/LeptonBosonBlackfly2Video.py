#!/usr/bin/env python2

import rospy
from cv_bridge import CvBridge
import rosbag
from sensor_msgs.msg import Image
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Combine Lepton, Boson, and Blackfly Data into a video.")
parser.add_argument("bagfile", help="Name of the bagfile used as input")
args = parser.parse_args()

fourcc = cv2.VideoWriter_fourcc(*'MPEG')

out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1920,  512))

bridge = CvBridge()

def ProcessLeptonMessage(msg):
    image = bridge.imgmsg_to_cv2(msg)
    clahe = cv2.createCLAHE(clipLimit=150.0, tileGridSize=(8,6))
    image = clahe.apply(image)
    image = (image / 256).astype(np.uint8)
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_NEAREST)
    return image


def ProcessBosonMessage(msg):
    image = bridge.imgmsg_to_cv2(msg)
    clahe = cv2.createCLAHE(clipLimit=150.0, tileGridSize=(20,16))
    image = clahe.apply(image)
    image = (image / 256).astype(np.uint8)
    print(image.shape)
    return image


def ProcessBlackflyMessage(msg):
    image = bridge.imgmsg_to_cv2(msg)
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_NEAREST)
    return image
    
    
def WriteImages(lepton_image, boson_image, blackfly_image):
    global out
    if((lepton_image is None) or (boson_image is None) or (blackfly_image is None)):
        return
    zeros = np.zeros((32, 640), dtype="uint8")
    lepton_image = np.vstack((lepton_image, zeros))
    blackfly_image = np.vstack((blackfly_image, zeros))
    full_image = np.hstack((lepton_image, boson_image, blackfly_image))
    full_image_color = cv2.cvtColor(full_image, cv2.COLOR_GRAY2BGR);
    cv2.imshow('output', full_image_color)
    cv2.waitKey(1)
    out.write(full_image_color)


def main():
    global out
    lepton_topic = "/lepton"
    boson_topic = "/boson_camera_array/cam_0/image_raw"
    blackfly_topic = "/camera_array/cam0/image_raw"

    lepton_image = None
    boson_image = None
    blackfly_image = None
    bag = rosbag.Bag(args.bagfile, 'r')
    
    lepton_flag = False
    boson_flag = False
    blackfly_flag = False
    
    for topic, msg, time, in bag.read_messages(topics=[lepton_topic, boson_topic, blackfly_topic]):
        image = None
        if(topic == lepton_topic):
            if(lepton_flag == True):
                WriteImages(lepton_image, boson_image, blackfly_image)
                lepton_flag = False
                boson_flag = False
                blackfly_flag = False
            lepton_image = ProcessLeptonMessage(msg)
            lepton_flag = True

            
        elif(topic == boson_topic):
            if(boson_flag == True):
                WriteImages(lepton_image, boson_image, blackfly_image)
                lepton_flag = False
                boson_flag = False
                blackfly_flag = False
            boson_image = ProcessBosonMessage(msg)
            
            
        elif(topic == blackfly_topic):
            if(blackfly_flag == True):
                WriteImages(lepton_image, boson_image, blackfly_image)
                lepton_flag = False
                boson_flag = False
                blackfly_flag = False
            blackfly_image = ProcessBlackflyMessage(msg)
            blackfly_flag = True
            
        else:
            print("WARNING: Read message from bad topic: " + str(topic))
    out.release()
        
	
if __name__ == "__main__":
    main()
        
