#!/usr/bin/env python2

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np

pub = rospy.Publisher('/lepton_processed', Image, queue_size=10)

CAMERA_MATRIX = np.array([[157.57555346, 0.00000000e+00, 79.06174387],
                          [0.00000000e+00, 157.6567718, 64.20473234],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
DISTORTION_COEFFICIENTS = np.array([-0.36583139,  0.34630364, -0.00260575,  0.00041044, -0.01098873])
LEPTON_RESOLUTION = (160, 120)
OUTPUT_MATRIX, OUTPUT_ROI = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DISTORTION_COEFFICIENTS, LEPTON_RESOLUTION, 1, LEPTON_RESOLUTION)

DO_UNDISTORT = True
DO_CONTRAST = True
DO_RESIZE = False

bridge = CvBridge()

def callback(data):
    image = bridge.imgmsg_to_cv2(data)
    processed_image = process_lepton_image(image)
    processed_data = bridge.cv2_to_imgmsg(processed_image, "mono8")
    pub.publish(processed_data)

def main():
    rospy.init_node('lepton_processor', anonymous=False)
    rospy.Subscriber("lepton", Image, callback)
    rospy.spin()

def process_lepton_image(image):
    if(DO_UNDISTORT):
        image = cv2.undistort(image, CAMERA_MATRIX, DISTORTION_COEFFICIENTS, None, OUTPUT_MATRIX)
        x, y, w, h = OUTPUT_ROI
        
        # Comment out this line to leave the undistorted area in the image
        #image = image[y:y+h, x:x+w]
    if(image.dtype == 'uint16'):
        if(DO_CONTRAST):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
        image = (image / 256).astype(np.uint8)
    if(DO_RESIZE):
        width = int(image.shape[1] / 3)
        height = int(image.shape[0] / 3)
        image = cv2.resize(image, (width, height))
    
    return image
	
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
        
