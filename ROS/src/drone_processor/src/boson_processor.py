#!/usr/bin/env python2

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np
from scipy import stats, spatial

pub = rospy.Publisher('/boson_processed_2', Image, queue_size=10)


CAMERA_MATRIX = np.array([[525.72910799, 0.00000000e+00, 304.93486687],
                          [0.00000000e+00, 525.58273357, 274.40590068],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
DISTORTION_COEFFICIENTS = np.array([-0.5305152, 0.56095799, -0.00274123, 0.00165569])

#[-0.59654054 ,  0.204940912, -0.08372369, 0.00832339]

BOSON_RESOLUTION = (640, 512)
OUTPUT_MATRIX, OUTPUT_ROI = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DISTORTION_COEFFICIENTS, BOSON_RESOLUTION, 1, BOSON_RESOLUTION)

DO_UNDISTORT = False
DO_CONTRAST = True
DO_RESIZE = False

bridge = CvBridge()

# Different contrasting methods can be implemented here
def Contrast16BitImg(image, low_bound, high_bound):
    image = image.astype('float')
    image -= low_bound
    image_negative_indicies = image < 0
    image[image_negative_indicies] = 0
    contrast_range = high_bound - low_bound
    image *= (255.0 / contrast_range)
    image_over_8bit_indicies = image > 255
    image[image_over_8bit_indicies] = 255
    image = np.rint(image)
    image = image.astype(np.uint8)
    return image

def callback(data):
    image = bridge.imgmsg_to_cv2(data)
    processed_image = process_boson_image(image)
    processed_data = bridge.cv2_to_imgmsg(processed_image, "mono8")
    pub.publish(processed_data)

def main():
    rospy.init_node('boson_processor', anonymous=False)
    rospy.Subscriber("/boson_camera_array/cam_0/image_raw", Image, callback)
    rospy.spin()

def process_boson_image(image):
    
    if(DO_UNDISTORT):
        image = cv2.undistort(image, CAMERA_MATRIX, DISTORTION_COEFFICIENTS, None, OUTPUT_MATRIX)
        x, y, w, h = OUTPUT_ROI
        
        # Comment out this line to leave the undistorted area in the image
        #image = image[y:y+h, x:x+w]
    if(image.dtype == 'uint16'):
        if(DO_CONTRAST):
            print(str(image.max()) + " | " + str(image.min()))
            img = image
            
            low_bound_range = (2**9)
            high_bound_range = (2**9)-1
            
            mode_array = stats.mode(img, axis=None)
            low_bound = (mode_array.mode[0]) - low_bound_range
            high_bound = (mode_array.mode[0]) + high_bound_range
            image = Contrast16BitImg(img, low_bound, high_bound)
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
        
