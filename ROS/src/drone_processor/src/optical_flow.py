#!/usr/bin/env python2

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import math
import numpy as np
import random
from math import sqrt
import time

pub = rospy.Publisher('/lepton_flow', Image, queue_size=10)

bridge = CvBridge()

random.seed(10)

'''
    Sparse Optical Flow Parameters
'''

# Number of features detected by sparse feature tracker
SPARSE_FEATURES = 2000

# Valid Methods: orb, random, orb_random, even
# orb uses Orb features to find sparse points
# random will randomly generate sparse points
# orb_random uses a combination of both orb features and randomly generated points for sparse features
# even will evenly distribute sparse points across the image
# If METHOD is not any of these, it will default to Orb
METHOD = "orb"

# Ratio of Orb to Random features detected. Only used if METHOD is orb_random
ORB_RANDOM_RATIO = 0.5

g_prev_image = None
g_prev_kp = None


def flow_ransac(flow, threshold_percent=0.1, min_inliers=500):
    if (flow.ndim != 3):
        raise Exception("Number of dimentions was not 3. Got " + str(flow.ndim))
    if (flow.shape[2] != 2):
        raise Exception("Invalid shape, must be (x,y,2). Got " + str(flow.shape))
    items = flow.shape[0] * flow.shape[1]
    vectors = flow.reshape(items, 2)

    solved_flag = False
    solved_vector = [0, 0]

    for i in random.sample(range(items), items):
        test_vector = vectors[i, :]
        solved_flag = False
        inliers = 0
        threshold = threshold_percent * sqrt((test_vector[0] ** 2) + (test_vector[1] ** 2))
        for j in range(items):
            measured_vector = vectors[j, :]

            if (sqrt(((test_vector[0] - measured_vector[0]) ** 2) + (
                    (test_vector[1] - measured_vector[1]) ** 2)) <= threshold):
                inliers += 1
            if (inliers > min_inliers):
                solved_flag = True
                solved_vector = test_vector
                break
        if (solved_flag == True):
            break

    return (solved_flag, solved_vector)


def flow_mode(flow, magnitude_bins=10, radial_bins=100, kernel=(3, 3)):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    largest = np.amax(mag)
    magnitude_bin_size = (largest / magnitude_bins) * 1.0000001
    radial_bin_size = 2 * np.pi / radial_bins
    ta = time.time()

    bins = np.zeros((magnitude_bins, radial_bins), dtype=int)
    vector_average_bins = np.zeros((magnitude_bins, radial_bins, 2))

    tb = time.time()

    for row in range(0, mag.shape[0], 2):
        for col in range(0, mag.shape[1], 2):
            magnitude_bin = int(mag[row, col] / magnitude_bin_size)
            radial_bin = int(ang[row, col] / radial_bin_size)
            vector_average_bins[magnitude_bin, radial_bin, 0] = ((vector_average_bins[magnitude_bin, radial_bin, 0] *
                                                                  bins[magnitude_bin, radial_bin]) + flow[
                                                                     row, col, 0]) / (
                                                                            bins[magnitude_bin, radial_bin] + 1)
            vector_average_bins[magnitude_bin, radial_bin, 1] = ((vector_average_bins[magnitude_bin, radial_bin, 1] *
                                                                  bins[magnitude_bin, radial_bin]) + flow[
                                                                     row, col, 1]) / (
                                                                            bins[magnitude_bin, radial_bin] + 1)
            bins[magnitude_bin, radial_bin] += 1

            # print(str(bins))
    tc = time.time()
    conv = np.zeros(((magnitude_bins - kernel[0] + 1), radial_bins))
    for row in range(1, conv.shape[0]):
        array_end = bins[row:row + kernel[0], -1]
        array_end = array_end.reshape(array_end.shape + (1,))
        array_begin = bins[row:row + kernel[0], 0:2]
        sub_mat = np.concatenate((array_end, array_begin), 1)
        conv[row, 0] = np.mean(sub_mat)
        for col in range(2, conv.shape[1]):
            col_offset = int(math.floor(kernel[1] / 2))
            sub_mat = bins[row:row + kernel[0], (col - col_offset):(col + kernel[1] - col_offset)]
            conv[row, col] = np.mean(sub_mat)

    # print(str(conv))

    td = time.time()
    largest_conv = np.argmax(conv)
    largest_conv_row = int(math.floor(largest_conv / conv.shape[1]))
    largest_conv_col = largest_conv % conv.shape[1]

    largest_bin = np.argmax(bins)
    largest_bin_col = int(math.floor(largest_bin / bins.shape[0]))
    largest_bin_row = largest_bin % bins.shape[0]

    filtered_vectors = []
    filtered_average_vectors = []
    num_filtered_average_vectors = 0
    for i in range(largest_conv_row - 1, largest_conv_row + 2):
        for j in range(largest_conv_col - 1, largest_conv_col + 2):
            while (i >= magnitude_bins):
                i -= magnitude_bins
            while (j >= radial_bins):
                j -= radial_bins
            filtered_average_vectors.append(list(vector_average_bins[i, j] * bins[i, j]))
            num_filtered_average_vectors += bins[i, j]
    te = time.time()

    np_array = np.array(filtered_average_vectors)

    avg_x = 0
    avg_y = 0
    if (np_array.size != 0):
        avg_x = sum(np_array[:, 0]) / num_filtered_average_vectors
        avg_y = sum(np_array[:, 1]) / num_filtered_average_vectors

    # print("Initialize time: " + str(tb - ta))
    # print("Bin time: " + str(tc - tb))
    # print("Conv time: " + str(td - tc))
    # print("Average time: " + str(te - td))

    # print("Bin max: " + str(bins.max()))
    # print("Bin location: " + str(np.argmax(bins)))
    # print("Conv max: " + str(conv.max()))
    # print("Conv location: " + str(largest_conv))
    # print(str(conv.shape[0]) + " | " + str(conv.shape[1]))

    # print("Avg  Estimation = (" + str(round(mag.sum() / mag.size, 2)) + ", " + str((ang.sum() / ang.size)*180/np.pi) + ")")
    # print("Bin  Estimation = (" + str(round(largest_bin_row * magnitude_bin_size, 2)) + ", " + str(largest_bin_col*3.6) + ")")
    # print("Conv Estimation = (" + str(round(largest_conv_row * magnitude_bin_size, 2)) + ", " + str(largest_conv_col*3.6) + ")")
    # print("Real Estimation = (" + str(avg_x) + ", " + str(avg_y) + ")")

    print((avg_x, avg_y))
    # print("Row: " + str(largest_conv_row))
    # print("Col: " + str(largest_conv_col))
    return (avg_x, avg_y)


def flow_to_image(flow, original_image):
    hsv_reference = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    hsv = np.zeros_like(hsv_reference)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # hsv[...,2] = mag*10
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr_flow


def dense_optical_flow(image):
    global g_prev_image
    if (not (g_prev_image is None)):
        flow = cv2.calcOpticalFlowFarneback(g_prev_image, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Remove any of the flow vectors where the image is black (typically borders due to camera calibration)
        img_mask = img > 1
        flow[:, :, 0] *= img_mask
        flow[:, :, 1] *= img_mask

        camera_vector = flow_mode(flow)
        total_x += camera_vector[0]

        total_y += camera_vector[1]
        #print(total_x)
        #print(total_y)

        flow_image = flow_to_image(flow, img)
        processed_data = bridge.cv2_to_imgmsg(flow_image, "bgr8")
        pub.publish(processed_data)
    g_prev_image = img


tracked_points = []

FEATURE_PARAMS = dict(maxCorners=SPARSE_FEATURES,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


def sparse_optical_flow(image):
    global g_prev_image
    global g_prev_kp
    global tracked_points

    good_new_points = []
    good_old_points = []

    if (not (g_prev_image is None)):
        updated_points, st, err = cv.calcOpticalFlowPyrLK(g_prev_image, image, g_prev_kp, None, **lk_params)

        if (updated_points is not None):
            good_new_points = updated_points[st == 1]
            good_old_points = tracked_points[st == 1]

        # Remove any of the flow vectors where the image is black (typically borders due to camera calibration)
        img_mask = img > 1
        flow[:, :, 0] *= img_mask
        flow[:, :, 1] *= img_mask

        camera_vector = flow_mode(flow)
        total_x += camera_vector[0]

        total_y += camera_vector[1]
        #print(total_x)
        #print(total_y)

        flow_image = flow_to_image(flow, img)
        processed_data = bridge.cv2_to_imgmsg(flow_image, "bgr8")
        pub.publish(processed_data)

    if (len(tracked_points) < SPARSE_FEATURES):
        tracked_points = cv.goodFeaturesToTrack(old_gray, mask=None, **FEATURE_PARAMS)

    g_prev_image = img


total_x = 0
total_y = 0


def callback(data):
    global g_prev_image
    global total_x
    global total_y
    img = bridge.imgmsg_to_cv2(data)
    if (not (g_prev_image is None)):
        flow = cv2.calcOpticalFlowFarneback(g_prev_image, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Remove any of the flow vectors where the image is black (typically borders due to camera calibration)
        img_mask = img > 1
        flow[:, :, 0] *= img_mask
        flow[:, :, 1] *= img_mask

        camera_vector = flow_mode(flow)
        total_x += camera_vector[0]

        total_y += camera_vector[1]
        #print("x = " + str(total_x))
        #print("y = " + str(total_y))

        flow_image = flow_to_image(flow, img)
        processed_data = bridge.cv2_to_imgmsg(flow_image, "bgr8")
        pub.publish(processed_data)
    g_prev_image = img


def main():
    rospy.init_node('optical_flow', anonymous=False)
    rospy.Subscriber("lepton_processed", Image, callback)
    rospy.spin()


if __name__ == "__main__":
    main()
