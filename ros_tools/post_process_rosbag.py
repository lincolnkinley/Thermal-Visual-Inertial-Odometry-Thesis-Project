#!/usr/bin/env python2

import rospy
from cv_bridge import CvBridge
import rosbag
from sensor_msgs.msg import Image
from std_msgs.msg import Time
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Combine Lepton, Boson, and Blackfly Data into a video.")
parser.add_argument("bagfile", help="Name of the bagfile used as input")
args = parser.parse_args()

bridge = CvBridge()

PROCESS_BLACKFLY = False

def ProcessLeptonMessage(msg):
    image = bridge.imgmsg_to_cv2(msg)
    clahe = cv2.createCLAHE(clipLimit=150.0, tileGridSize=(8,6))
    image = clahe.apply(image)
    image = (image / 256).astype(np.uint8)
    message = bridge.cv2_to_imgmsg(image, "mono8")
    message.header = msg.header
    return message


def ProcessBosonMessage(msg):
    image = bridge.imgmsg_to_cv2(msg)
    clahe = cv2.createCLAHE(clipLimit=150.0, tileGridSize=(40,32))
    image = clahe.apply(image)
    image = (image / 256).astype(np.uint8)
    message = bridge.cv2_to_imgmsg(image, "mono8")
    message.header = msg.header
    return message
    
def ProcessBlackflyMessage(msg):
    image = bridge.imgmsg_to_cv2(msg)
    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(40,30))
    image = clahe.apply(image)
    image = cv2.GaussianBlur(image,(3,3),0)
    message = bridge.cv2_to_imgmsg(image, "mono8")
    message.header = msg.header
    return message
    
    

def main():
    lepton_in_topic = "/lepton"
    boson_in_topic = "/boson_camera_array/cam_0/image_raw"
    blackfly_topic = "/camera_array/cam0/image_raw"
    
    lepton_out_topic = "/lepton_processed"
    boson_out_topic = "/boson_processed"
    blackfly_out_topic = "/blackfly_processed"
    
    print("Opening rosbags...")
    read_bag = rosbag.Bag(args.bagfile, 'r')
    write_bag = rosbag.Bag(args.bagfile[:-4] + "_processed.bag", 'w')
    print("rosbags opened!")
    
    blackfly_flag = False
    last_blackfly_time = None
    
    loop = 0
    pct = 0
    
    bag_size = read_bag.get_message_count()
    
    prev_time = None
    
    for topic, msg, time, in read_bag.read_messages():
        if(topic == lepton_in_topic):
            write_bag.write(topic, msg, msg.header.stamp)
            msg = ProcessLeptonMessage(msg)
            topic = lepton_out_topic

        elif(topic == boson_in_topic):
            write_bag.write(topic, msg, msg.header.stamp)
            msg = boson_image = ProcessBosonMessage(msg)
            if((blackfly_flag == True) and (abs((msg.header.stamp - last_blackfly_time).to_sec()) <= 0.03) and not(last_blackfly_time is None)):
                msg.header.stamp = last_blackfly_time
                blackfly_flag = False
                #print("Boson aligned!")
            else:
                if(last_blackfly_time is not None):
                    print("Boson failed to align: " + str(blackfly_flag) + " | " + str(abs(msg.header.stamp - last_blackfly_time)) + " | " + str(last_blackfly_time))
                else:
                    print("Boson failed to align: " + str(blackfly_flag) + " | " + str(last_blackfly_time))
                
                msg.header.stamp = msg.header.stamp - rospy.Duration(0.020657)
            topic = boson_out_topic
            
        elif(topic == blackfly_topic):
            last_blackfly_time = msg.header.stamp
            blackfly_flag = True
            if(PROCESS_BLACKFLY):
                write_bag.write(topic, msg, msg.header.stamp)
                msg = ProcessBlackflyMessage(msg)
                topic = blackfly_out_topic
        
        if topic == "/tf" and msg.transforms:
            outbag.write(topic, msg, msg.transforms[0].header.stamp)
        elif(msg._has_header):
            write_bag.write(topic, msg, msg.header.stamp)
            prev_time = msg.header.stamp
        elif(prev_time is not None):
            write_bag.write(topic, msg, prev_time)
        else:
            print("WARNGING! Dropped a message from topic " + topic + " because time could not be determined!")
        
        #print(str(loop) + " | " + str(bag_size) + " | " + str(pct))
        
        if((100*float(loop))/float(bag_size) > float(pct)):
            print(str(pct) + "% complete")
            pct += 1
        
        loop += 1
        
    read_bag.close()
    write_bag.close()
        
	
if __name__ == "__main__":
    main()
        
