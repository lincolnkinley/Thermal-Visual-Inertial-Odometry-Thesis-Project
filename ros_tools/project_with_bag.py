#!/usr/bin/env python2

import rospy
from cv_bridge import CvBridge
import rosbag
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
import cv2
import argparse
import numpy as np
from tf.transformations import quaternion_matrix, quaternion_from_matrix, quaternion_from_euler, euler_from_quaternion
import math

parser = argparse.ArgumentParser(description="Break rosbag into images")
parser.add_argument("bagfile", help="Name of the bagfile used as input")
args = parser.parse_args()

bridge = CvBridge()

BOSON_MATRIX = np.array([[525.72910799, 0.00000000e+00, 304.93486687],
                          [0.00000000e+00, 525.58273357, 274.40590068],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
BOSON_DISTORTION = np.array([-0.5305152, 0.56095799, -0.00274123, 0.00165569])
BOSON_RESOLUTION = (640, 512)
BOSON_OUTPUT_MATRIX, BOSON_OUTPUT_ROI = cv2.getOptimalNewCameraMatrix(BOSON_MATRIX, BOSON_DISTORTION, BOSON_RESOLUTION, 1, BOSON_RESOLUTION)


LEPTON_MATRIX = np.array([[157.57555346, 0.00000000e+00, 79.06174387],
                          [0.00000000e+00, 157.6567718, 64.20473234],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
LEPTON_DISTORTION = np.array([-0.36583139,  0.34630364, -0.00260575,  0.00041044])
LEPTON_RESOLUTION = (160, 120)
LEPTON_OUTPUT_MATRIX, LEPTON_OUTPUT_ROI = cv2.getOptimalNewCameraMatrix(LEPTON_MATRIX, LEPTON_DISTORTION, LEPTON_RESOLUTION, 1, LEPTON_RESOLUTION)

BLACKFLY_MATRIX = np.array([[882.8608306106577, 0.00000000e+00, 388.74615057267823],
                           [0.00000000e+00, 883.678200169175, 276.5905850730814],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
BLACKFLY_DISTORTION = np.array([-0.19932066859796568, 0.12623897725827676, 0.00043605530205857403, 0.0014082317201669727])
BLACKFLY_RESOLUTION = (720, 540)
BLACKFLY_OUTPUT_MATRIX, BLACKFLY_OUTPUT_ROI = cv2.getOptimalNewCameraMatrix(BLACKFLY_MATRIX, BLACKFLY_DISTORTION, BLACKFLY_RESOLUTION, 1, BLACKFLY_RESOLUTION)


def ProcessBosonMessage(msg):
    image = bridge.imgmsg_to_cv2(msg)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(40,32))
    image = clahe.apply(image)
    image = (image / 256).astype(np.uint8)
    #image = cv2.GaussianBlur(image,(3,3),0)
    #image = cv2.equalizeHist(image)
    #image = cv2.fastNlMeansDenoising(image, 8, 8, 7, 15)
    #image = cv2.bilateralFilter(image,9,5,5)
    return image


PROCESS_BLACKFLY = False

def RS_Show(image, name="image"):
    if(image.shape[0] > 1080):
        scale = 1080.0 / float(image.shape[0])
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    if(image.shape[1] > 1920):
        scale = 1920.0 / float(image.shape[1])
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow(name, image)
    k = cv2.waitKey(10)

def main():
    lepton_topic = "/lepton_processed"
    boson_topic = "/boson_camera_array/cam_0/image_raw"
    blackfly_topic = "/camera_array/cam0/image_raw"
    imu_topic = "/imu/imu"
    
    if(PROCESS_BLACKFLY == True):
        blackfly_topic = "/blackfly_processed"
    
    print("Opening rosbags...")
    read_bag = rosbag.Bag(args.bagfile, 'r')
    print("rosbags opened!")
    
    topics_to_read = [lepton_topic, imu_topic]
    bag_size = read_bag.get_message_count(topics_to_read)
    
    lepton_number = 0
    boson_number = 0
    blackfly_number = 0
    
    loop = 0
    pct = 0
    
    '''
    # Estimate
    static_transform = np.array([[ 0.7071068, 0.0000000, -0.7071068], 
                                 [0.0000000,  1.0000000,  0.0000000], 
                                 [0.7071068,  0.0000000,  0.7071068]])
    # Kalibr                   
    static_transform = np.array([[-0.00900976, -0.99987985, -0.01261378], 
                                 [ 0.73181808, -0.01518949,  0.68133074],
                                 [-0.68144048, -0.00309237,  0.731867  ]])
    
    # Projection
    static_transform = np.array([[2.0, 0.0,    -640.0],
                                 [0.0, 2,      -512.0],
                                 [0.0, 0.00195, 1.0]])
    '''
   
    boson_skip_factor = 12
    blackfly_skip_factor = 12
    lepton_skip_factor = 4
    
    boson_imgs = boson_skip_factor
    blackfly_imgs = blackfly_skip_factor
    lepton_imgs = lepton_skip_factor
   
    quat = None 
    zzz = 0.0  
    
    for topic, msg, time, in read_bag.read_messages(topics=topics_to_read):
        
        if((100*float(loop))/float(bag_size) > float(pct)):
            print(str(pct) + "% complete")
            pct += 1
        
        loop += 1
        
        if(topic == lepton_topic):
            lepton_imgs += 1
            if(lepton_imgs <= lepton_skip_factor or quat is None):
                continue
            lepton_imgs = 0
            filename = "lepton_day_complete/{number:06}.png".format(number=lepton_number)
            image = bridge.imgmsg_to_cv2(msg)
            image = cv2.undistort(image, LEPTON_MATRIX, LEPTON_DISTORTION, None, LEPTON_OUTPUT_MATRIX)
            lepton_number += 1
            #cv2.imwrite(filename, image)
            static_transform = np.array([[2.0, 0.0,    -160.0],
                                         [0.0, 2,      -120.0],
                                         [0.0, 0.00835, 1.0]])

            
        elif(topic == boson_topic):
            boson_imgs += 1
            if(boson_imgs <= boson_skip_factor or quat is None):
                continue
            if(abs(quat.x) > 0.05 or abs(quat.y) > 0.05):
                continue
            boson_imgs = 0
            filename = "boson/{number:06}.png".format(number=boson_number)
            image = ProcessBosonMessage(msg)
            image = cv2.undistort(image, BOSON_MATRIX, BOSON_DISTORTION, None, BOSON_OUTPUT_MATRIX)
            boson_number += 1
            #cv2.imwrite(filename, image)
            static_transform = np.array([[2.0, 0.0,    -640.0],
                                         [0.0, 2,      -512.0],
                                         [0.0, 0.00195, 1.0]])
                                         
            static_transform = np.array([[2.0, 0.0,    -640.0],
                                         [0.0, 2,      -512.0],
                                         [0.0, 0.006, 1.0]])
            
        elif(topic == blackfly_topic):
            blackfly_imgs += 1
            if(blackfly_imgs <= blackfly_skip_factor or quat is None):
                continue
            blackfly_imgs = 0
            filename = "blackfly/{number:06}.png".format(number=blackfly_number)
            image = bridge.imgmsg_to_cv2(msg)
            image = cv2.undistort(image, BLACKFLY_MATRIX, BLACKFLY_DISTORTION, None, BLACKFLY_OUTPUT_MATRIX)
            blackfly_number += 1
            #cv2.imwrite(filename, image)
            static_transform = np.array([[2.0, 0.0,    -720.0],
                                         [0.0, 2,      -540.0],
                                         [0.0, 0.00185, 1.0]])
            
        elif(topic == imu_topic):
            quat = msg.orientation
            imu_orientation = quaternion_matrix([quat.x, quat.y, quat.z, quat.w,])[:3,:3]
            rpy = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w,])
            nox_quat = quaternion_from_euler(rpy[0], rpy[1], 0)
            
            quat.x = nox_quat[0]
            quat.y = nox_quat[1]
            quat.z = nox_quat[2]
            quat.w = nox_quat[3]
            continue
            
        else:
            print("Warning! Unexpected topic received: " + topic)
            continue
           
        ''' 
        r = math.pi/4
        
        transform = np.array([[2.229, 3.2, 0.0],
                              [0.0, 6.9333, 0.0],
                              [0.0, 0.0043244+zzz, 1.0]])
                              
        transform = np.array([[2.0, 0.0, -640.0],
                              [0.0, 2, -512.0],
                              [0.0, 0.00195, 1.0]])
                         
moving forward
quat.x = 0.00688279792666
quat.y = 0.0308335479349
imu_x = 0.000107543717604
imu_y = -0.000604579371272

quat.x = 0.00947204977274
quat.y = 0.031170476228
imu_x = -4.73602488637e-05
imu_y = 0.00031170476228


stopping
quat.x = -0.00462101353332
quat.y = -0.0574850402772
imu_x = -7.22033364582e-05
imu_y = 0.00112715765249

quat.x = -0.00356178311631
quat.y = -0.072200499475
imu_x = 1.78089155816e-05
imu_y = -0.00072200499475



                              
        '''
            
        h, w = image.shape[:2]
        
        #print("quat.x = " + str(quat.x))
        #print("quat.y = " + str(quat.y))
                                  
        imu_transform = np.array([[1.0/(1.0+(0*quat.x)), 0.0, -h/2],
                                  [0.0, 1.0/(1.0+(0*quat.y)), -w/2],
                                  [-quat.x / w, -quat.y / h, 1.0]])
                          
        transform = np.matmul(imu_transform, static_transform)
       

        
        #transform = imu_transform
        #transform = static_transform
        
        corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners_aft = cv2.perspectiveTransform(corners_bef, transform)
        xmin = math.floor(corners_aft[:, 0, 0].min())
        ymin = math.floor(corners_aft[:, 0, 1].min())
        xmax = math.ceil(corners_aft[:, 0, 0].max())
        ymax = math.ceil(corners_aft[:, 0, 1].max())
        translate = np.eye(3)
        translate[0, 2] = -xmin
        translate[1, 2] = -ymin

        transform = np.matmul(translate, transform)
        
        a = int(math.ceil(xmax - xmin))
        b = int(math.ceil(ymax - ymin))
        warped_image = cv2.warpPerspective(image, transform, (a, b))
        #RS_Show(warped_image)
        cv2.imwrite(filename, warped_image)

        

        

    read_bag.close()
        
	
if __name__ == "__main__":
    main()
        
