#!/usr/bin/env python2

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np

pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)

CAMERA_MATRIX = np.array([[1.75093888e+03, 0.00000000e+00, 7.48018258e+02],
                          [0.00000000e+00, 1.75407471e+03, 5.83197350e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
DISTORTION_COEFFICIENTS = np.array([-0.2029467, 0.10808601, 0.00550694, 0.0010867, 0.63987819])
SPINNAKER_RESOLUTION = (1440, 1080)
OUTPUT_MATRIX, OUTPUT_ROI = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DISTORTION_COEFFICIENTS, SPINNAKER_RESOLUTION, 1, SPINNAKER_RESOLUTION)

DO_UNDISTORT = False
DO_RESIZE = True

bridge = CvBridge()

def callback(data):
    image = bridge.imgmsg_to_cv2(data)
    processed_image = process_spinnaker_image(image)
    processed_data = bridge.cv2_to_imgmsg(processed_image, "mono8")
    pub.publish(processed_data)

def main():
    rospy.init_node('spinnaker_processor', anonymous=False)
    rospy.Subscriber("/camera_array/cam0/image_raw", Image, callback)
    rospy.spin()

def process_spinnaker_image(image):
    if(DO_UNDISTORT):
        image = cv2.undistort(image, CAMERA_MATRIX, DISTORTION_COEFFICIENTS, None, OUTPUT_MATRIX)
        x, y, w, h = OUTPUT_ROI
        image = undistorted[y:y+h, x:x+w]
    if(DO_RESIZE):
        width = int(image.shape[1] / 9)
        height = int(image.shape[0] / 9)
        image = cv2.resize(image, (width, height))
    return image
	
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
        
