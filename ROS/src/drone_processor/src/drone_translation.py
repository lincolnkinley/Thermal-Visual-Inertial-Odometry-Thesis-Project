#!/usr/bin/env python2
import time

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
from time import sleep
import math

pub = rospy.Publisher('/lepton_odom', Odometry, queue_size=10)
img_pub = rospy.Publisher('/tracked_features', Image, queue_size=10)

prev_points_id = []
prev_points_valid = False
prev_points = np.empty((0, 2), dtype="float")

current_points = np.array([], dtype="float")
current_points_valid = False

published_rgb_img_valid = False
published_rgb_img = np.zeros((160, 120, 3), dtype="uint8")

bridge = CvBridge()

def img_callback(data):
    global published_rgb_img, published_rgb_img_valid
    published_rgb_img_valid = False
    image = bridge.imgmsg_to_cv2(data)
    published_rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    published_rgb_img_valid = True
    return


    while(current_points_valid == False):
        time.sleep(0.0001)
    for i in range(current_points.shape[0]):
        rbg_image = cv2.circle(rbg_image, tuple(current_points[i, :]), radius=3, color=(0, 0, 255), thickness=2)

    while(prev_points_valid == False):
        time.sleep(0.0001)
    for i in range(prev_points.shape[0]):
        rbg_image = cv2.circle(rbg_image, tuple(prev_points[i, :].astype(int)), radius=4, color=(0, 255, 0), thickness=1)

    img_msg = bridge.cv2_to_imgmsg(rbg_image, "bgr8")
    img_pub.publish(img_msg)

def points_callback(data):
    global prev_points, prev_points_id, current_points, current_points_valid, prev_points_valid
    current_points_valid = False
    prev_points_valid = False


    channels = data.channels
    # channels [point_id, x, y, x_velocity, y_velocity], velocity doesn't work

    points_id = channels[0].values
    points_x = channels[1].values
    points_y =  channels[2].values

    points = np.vstack((points_x, points_y))
    points = np.transpose(points)

    points, points_id = filter_points(points, prev_points, points_id, [160, 120], 5)

    current_points = points.astype(int)
    current_points_valid = True

    both_id = np.intersect1d(points_id, prev_points_id)

    current_matches = np.float32([points[points_id.index(i), :] for i in both_id])
    previous_matches = np.float32([prev_points[prev_points_id.index(i), :] for i in both_id])

    '''
    print(points_id)
    print(points)
    print("--------------------------------------------------------")
    print(prev_points_id)
    print(prev_points)
    print("--------------------------------------------------------")
    print(current_matches)
    print(previous_matches)
    '''
    #print("########################################################")


    if(len(points_x) != len(points_y)):
        print("WARNING! x and y dimensions are invalid")

    if((points.shape[0] >= 4) and (prev_points.shape[0] >= 4)):
        try:
            H, H_inliers = cv2.findHomography(current_matches, previous_matches, method=cv2.LMEDS)
            #print(len(current_matches))
            #print(H)
        except Exception as E:
            pass
            #print(E)

    while(published_rgb_img_valid == False):
        time.sleep(0.001)

    rgb_image = published_rgb_img
    for i in range(current_matches.shape[0]):
        rgb_image = cv2.line(rgb_image, tuple(current_matches[i, :].astype(int)), tuple(previous_matches[i, :].astype(int)), color=(255, 255, 0))

    for i in range(current_points.shape[0]):
        rgb_image = cv2.circle(rgb_image, tuple(points[i, :].astype(int)), radius=3, color=(0, 0, 255), thickness=2)

    for i in range(prev_points.shape[0]):
        rgb_image = cv2.circle(rgb_image, tuple(prev_points[i, :].astype(int)), radius=4, color=(0, 255, 0),
                               thickness=1)

    img_msg = bridge.cv2_to_imgmsg(rgb_image, "bgr8")
    img_pub.publish(img_msg)

    prev_points_id = points_id
    prev_points = points
    prev_points_valid = True

def point_distance(a, b):
    return math.sqrt(((a[0]-b[0])**2) + ((a[1]-b[1])**2))

def filter_points(current_points, previous_points, points_id, image_shape, border):
    filtered_points = current_points
    for i in range(filtered_points.shape[0]):
        if((filtered_points[i,0] < border) or (image_shape[0] - border < filtered_points[i,0]) or (filtered_points[i,1] < border) or (image_shape[1] - border < filtered_points[i,1])):
            # Point close to the borders
            filtered_points[i, :] = [0, 0]
            continue
    idx = np.argwhere(np.all(filtered_points[..., :] == 0, axis=1))
    filtered_points = np.delete(filtered_points, idx, axis=0)
    filtered_points_id = np.delete(points_id, idx, axis=0)
    return filtered_points, list(filtered_points_id)

def main():
    rospy.init_node('lepton_odom', anonymous=False)
    rospy.Subscriber("feature_tracker/feature", PointCloud, points_callback)
    rospy.Subscriber("feature_tracker/raw_image", Image, img_callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    
