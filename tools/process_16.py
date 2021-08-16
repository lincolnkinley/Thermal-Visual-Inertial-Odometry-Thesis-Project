import cv2
import argparse
import os
import numpy
import math
from scipy import stats, spatial

parser = argparse.ArgumentParser()

parser.add_argument("-dir", type=str, dest="input_folder", required=True, help="Directory used for input.")

args = parser.parse_args()




def bounding_box(points, min_x=-numpy.inf, max_x=numpy.inf, min_y=-numpy.inf,
                 max_y=numpy.inf):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------
    points: (n,2) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1],
                ...,
                [xn,yn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keept or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = numpy.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = numpy.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)

    bb_filter = numpy.logical_and(bound_x, bound_y)

    return bb_filter


def tiled_features(kp, img_shape, tiley, tilex):
    '''
    Given a set of keypoints, this divides the image into a grid and returns
    len(kp)/(tilex*tiley) maximum responses within each tell. If that cell doesn't
    have enough points it will return all of them.
    '''
    feat_per_cell = int(len(kp) / (tilex * tiley))
    HEIGHT, WIDTH = img_shape
    assert WIDTH % tiley == 0, "Width is not a multiple of tilex"
    assert HEIGHT % tilex == 0, "Height is not a multiple of tiley"
    w_width = int(WIDTH / tiley)
    w_height = int(HEIGHT / tilex)

    xx = numpy.linspace(0, HEIGHT - w_height, tilex, dtype='int')
    yy = numpy.linspace(0, WIDTH - w_width, tiley, dtype='int')

    kps = numpy.array([])
    pts = numpy.array([keypoint.pt for keypoint in kp])
    kp = numpy.array(kp)

    for ix in xx:
        for iy in yy:
            inbox_mask = bounding_box(pts, iy, iy + w_width, ix, ix + w_height)
            inbox = kp[inbox_mask]
            inbox_sorted = sorted(inbox, key=lambda x: x.response, reverse=True)
            inbox_sorted_out = inbox_sorted[:feat_per_cell]
            kps = numpy.append(kps, inbox_sorted_out)
    return kps.tolist()


def radial_non_max(kp_list, dist):
    '''
    Given a set of Keypoints this finds the maximum response within radial
    distance from each other
    '''
    kp = numpy.array(kp_list)
    kp_mask = numpy.ones(len(kp), dtype=bool)
    pts = [k.pt for k in kp]
    tree = spatial.cKDTree(pts)
    # print ("len of kp1:",len(kp))
    for i, k in enumerate(kp):
        if kp_mask[i]:
            pt = tree.data[i]
            idx = tree.query_ball_point(tree.data[i], dist, p=2., eps=0, n_jobs=1)
            resp = [kp[ii].response for ii in idx]
            _, maxi = max([(v, i) for i, v in enumerate(resp)])
            del idx[maxi]
            for kp_i in idx:
                kp_mask[kp_i] = False
    return kp[kp_mask].tolist()





def Contrast16BitImg(image, low_bound, high_bound):
    image = image.astype('float')
    image -= low_bound
    image_negative_indicies = image < 0
    image[image_negative_indicies] = 0
    contrast_range = high_bound - low_bound
    image *= (255.0 / contrast_range)
    image_over_8bit_indicies = image > 255
    image[image_over_8bit_indicies] = 255
    image = numpy.rint(image)
    image = image.astype(numpy.uint8)
    return image

def SiftFeatureImage(image):
    height = image.shape[0]
    width = image.shape[1]

    sift = cv2.SIFT_create(1000, 3, 0.04, 10, 1.6)
    kp, des1 = sift.detectAndCompute(image, None)

    output = numpy.zeros((int(height), int(width)), dtype=numpy.uint8)
    if not kp:
        return output

    better_corners = kp
    #better_corners = tiled_features(kp, (width, height), 4, 4)
    #better_corners = radial_non_max(kp, 2)

    better_points = [better_point.pt for better_point in better_corners]
    for point in better_points:
        output[int(point[1]), int(point[0])] = 255
    corners_detected = numpy.count_nonzero(output)
    cv2.putText(output, str(corners_detected), (4, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 200, 2)
    return output

def OrbFeatureImage(image):
    height = image.shape[0]
    width = image.shape[1]

    orb = cv2.ORB_create(1000, 2, 3, 10, 0, 2, cv2.ORB_HARRIS_SCORE, 31, )
    kp, des1 = orb.detectAndCompute(image, None)

    output = numpy.zeros((int(height), int(width)), dtype=numpy.uint8)
    if not kp:
        return output

    better_corners = kp
    #better_corners = tiled_features(kp, (width, height), 4, 4)
    #better_corners = radial_non_max(kp, 2)

    better_points = [better_point.pt for better_point in better_corners]
    for point in better_points:
        output[int(point[1]), int(point[0])] = 255
    corners_detected = numpy.count_nonzero(output)
    cv2.putText(output, str(corners_detected), (4, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 200, 2)
    return output

def HarrisCorners(image):
    corners = cv2.cornerHarris(image, 2, 3, 0.04)
    height = image.shape[0]
    width = image.shape[1]
    output = numpy.zeros((int(height), int(width)), dtype=numpy.uint8)

    kp = []
    sorted_corners = sorted(corners.flatten())
    thresh = sorted_corners[-1001]
    for i in range(height):
        for j in range(width):
            value = corners[i,j]
            if(value > thresh):
                kp.append(cv2.KeyPoint(j, i, 13))
    max = corners.max()
    if not kp:
        return output

    better_corners = kp
    #better_corners = tiled_features(kp, (width, height), 4, 4)
    #better_corners= radial_non_max(kp, 2)

    better_points = [better_point.pt for better_point in better_corners]
    for point in better_points:
        output[int(point[1]), int(point[0])] = 255
    corners_detected = numpy.count_nonzero(output)
    cv2.putText(output, str(corners_detected), (4, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 200, 2)
    return output


def SobelEdge(image):
    # Taken from https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    blur_image = cv2.GaussianBlur(image, (3, 3), 0)

    grad_x = cv2.Sobel(blur_image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta,
                       borderType=cv2.BORDER_DEFAULT)

    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(blur_image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta,
                       borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad


def Histogram(image):
    # Define some constants for the graph size
    FULL_GRAPH_SIZE = 320
    GRAPH_SIZE = 256
    BIN_SIZE = 4

    GRAPH_COLOR = 20
    BACKGROUND_COLOR = 255
    AXIS_COLOR = 210
    GRID_COLOR = 120

    # create the histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, GRAPH_SIZE])
    hist_percentages = (hist * 100.0) / image.size

    # Create the binned array
    number_of_bins = int(GRAPH_SIZE / BIN_SIZE)
    binned_hist = numpy.zeros(number_of_bins)

    # Add the hist_percentages to bins
    for i in range(hist.size):
        bin_offset = int(math.floor(i/BIN_SIZE))
        binned_hist[bin_offset] += hist_percentages[i][0]

    # Create the graph image
    graph = numpy.full((FULL_GRAPH_SIZE,FULL_GRAPH_SIZE), BACKGROUND_COLOR, dtype=numpy.uint8)

    # Add space for the axis
    graph[GRAPH_SIZE:, :] = AXIS_COLOR
    graph[:, :FULL_GRAPH_SIZE-GRAPH_SIZE] = AXIS_COLOR

    # Fill in the graph
    for i in range(len(binned_hist)):
        y_on_graph = GRAPH_SIZE - int(round(binned_hist[i]*(256.0/100.0)))
        bottom_of_graph = GRAPH_SIZE
        x_on_graph = i*BIN_SIZE + FULL_GRAPH_SIZE - GRAPH_SIZE
        graph[y_on_graph:bottom_of_graph, x_on_graph:(x_on_graph + BIN_SIZE)] = GRAPH_COLOR

    H_ROWS = 10
    for i in range(H_ROWS+1):
        graph[int(i * GRAPH_SIZE/(H_ROWS)), :] = GRID_COLOR
        cv2.putText(graph, str(int(i * 100 / (H_ROWS))), (4, int((H_ROWS - i) * GRAPH_SIZE / (H_ROWS) + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    0,
                    2)

    V_ROWS = 8
    for i in range(V_ROWS + 1):
        graph[:, int(i * GRAPH_SIZE/(V_ROWS)) + 63] = GRID_COLOR
        cv2.putText(graph, str(int(i*(GRAPH_SIZE/V_ROWS))), (int((i * GRAPH_SIZE/(V_ROWS))+32), FULL_GRAPH_SIZE-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)

    return graph


def main():
    directory = args.input_folder
    i = 0
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # 'x264' doesn't work
    video = cv2.VideoWriter(directory + 'video.mp4', fourcc, 9, (1920, 960))
    for filename in sorted(os.listdir(directory)):
        if (filename.endswith(".png")):
            img = cv2.imread(directory + filename, -1)

            low_bound_range = (2**9)
            high_bound_range = (2**9)-1

            # Fixed Contrast (Doesn't work great)
            fixed_low_bound = 28000
            fixed_high_bound = 30000
            fixed_img = Contrast16BitImg(img, fixed_low_bound, fixed_high_bound)

            # Automatic contrast based on the image (low noise but sudden changes cause brightness shifts)
            low_bound = numpy.min(img)
            high_bound = numpy.max(img)
            auto_img = Contrast16BitImg(img, low_bound, high_bound)

            cv2.imwrite(str(directory) + "auto/" + filename, auto_img)

            # Average and fixed range (In between automatic and median on good images, but images with large areas of high contrast become a white and black blob)
            low_bound = numpy.average(img) - low_bound_range
            high_bound = numpy.average(img) + high_bound_range
            avg_img = Contrast16BitImg(img, low_bound, high_bound)

            # Median and fixed range (seemingly no brightness shift but high noise)
            low_bound = numpy.median(img) - low_bound_range
            high_bound = numpy.median(img) + high_bound_range
            med_img = Contrast16BitImg(img, low_bound, high_bound)

            cv2.imwrite(str(directory) + "processed/" + filename, med_img )

            # Mode and fixed range (Decent contrast but high noise and brightness flashes)
            mode_array = stats.mode(img, axis=None)
            low_bound = (mode_array.mode[0]) - low_bound_range
            high_bound = (mode_array.mode[0]) + high_bound_range
            mode_img = Contrast16BitImg(img, low_bound, high_bound)

            # CLAHE Method
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(img)
            clahe_img = (clahe_img / 256).astype(numpy.uint8)

            cv2.imwrite(str(directory) + "clahe/" + filename, clahe_img)

            fixed_sobel = SobelEdge(fixed_img)
            auto_sobel = SobelEdge(auto_img)
            avg_sobel = SobelEdge(avg_img)
            med_sobel = SobelEdge(med_img)
            mode_sobel = SobelEdge(mode_img)
            clahe_sobel = SobelEdge(clahe_img)

            fixed_orb = OrbFeatureImage(fixed_img)
            auto_orb = OrbFeatureImage(auto_img)
            avg_orb = OrbFeatureImage(avg_img)
            med_orb = OrbFeatureImage(med_img)
            mode_orb = OrbFeatureImage(mode_img)
            clahe_orb = OrbFeatureImage(clahe_img)

            fixed_sift = SiftFeatureImage(fixed_img)
            auto_sift = SiftFeatureImage(auto_img)
            avg_sift = SiftFeatureImage(avg_img)
            med_sift = SiftFeatureImage(med_img)
            mode_sift = SiftFeatureImage(mode_img)
            clahe_sift = SiftFeatureImage(clahe_img)

            fixed_harris = HarrisCorners(fixed_img)
            auto_harris = HarrisCorners(auto_img)
            avg_harris = HarrisCorners(avg_img)
            med_harris = HarrisCorners(med_img)
            mode_harris = HarrisCorners(mode_img)
            clahe_harris = HarrisCorners(clahe_img)

            fixed_hist = Histogram(fixed_img)
            auto_hist = Histogram(auto_img)
            avg_hist = Histogram(avg_img)
            med_hist = Histogram(med_img)
            mode_hist = Histogram(mode_img)
            clahe_hist = Histogram(clahe_img)

            width = int(320)
            height = int(240)

            fixed_img = cv2.resize(fixed_img, (width, height))
            auto_img = cv2.resize(auto_img, (width, height))
            avg_img = cv2.resize(avg_img, (width, height))
            med_img = cv2.resize(med_img, (width, height))
            mode_img = cv2.resize(mode_img, (width, height))
            clahe_img = cv2.resize(clahe_img, (width, height))

            fixed_sobel = cv2.resize(fixed_sobel, (width, height))
            auto_sobel = cv2.resize(auto_sobel, (width, height))
            avg_sobel = cv2.resize(avg_sobel, (width, height))
            med_sobel = cv2.resize(med_sobel, (width, height))
            mode_sobel = cv2.resize(mode_sobel, (width, height))
            clahe_sobel = cv2.resize(clahe_sobel, (width, height))

            fixed_orb = cv2.resize(fixed_orb, (width, height))
            auto_orb = cv2.resize(auto_orb, (width, height))
            avg_orb = cv2.resize(avg_orb, (width, height))
            med_orb = cv2.resize(med_orb, (width, height))
            mode_orb = cv2.resize(mode_orb, (width, height))
            clahe_orb = cv2.resize(clahe_orb, (width, height))

            fixed_sift = cv2.resize(fixed_sift, (width, height))
            auto_sift = cv2.resize(auto_sift, (width, height))
            avg_sift = cv2.resize(avg_sift, (width, height))
            med_sift = cv2.resize(med_sift, (width, height))
            mode_sift = cv2.resize(mode_sift, (width, height))
            clahe_sift = cv2.resize(clahe_sift, (width, height))

            fixed_harris = cv2.resize(fixed_harris, (width, height))
            auto_harris = cv2.resize(auto_harris, (width, height))
            avg_harris = cv2.resize(avg_harris, (width, height))
            med_harris = cv2.resize(med_harris, (width, height))
            mode_harris = cv2.resize(mode_harris, (width, height))
            clahe_harris = cv2.resize(clahe_harris, (width, height))

            '''
            #fixed_hist_plot = Histogram(fixed_img)

            #fig = fixed_hist_plot.figure()
            fig = plt.hist(fixed_img.ravel(), 256, [0, 256])
            fig.canvas.draw()
            data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
            fixed_hist = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            #auto_hist_plot = Histogram(auto_img)
            #fig = auto_hist_plot.figure()
            fig.canvas.draw()
            data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
            auto_hist = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            #avg_hist_plot = Histogram(avg_img)
            #fig = avg_hist_plot.figure()
            fig.canvas.draw()
            data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
            avg_hist = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            #med_hist_plot = Histogram(med_img)
            #fig = med_hist_plot.figure()
            fig.canvas.draw()
            data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
            med_hist = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            #mode_hist_plot = Histogram(mode_img)
            #fig = mode_hist_plot.figure()
            fig.canvas.draw()
            data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
            mode_hist = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            #clahe_hist_plot = Histogram(clahe_img)
            #fig = clahe_hist_plot.figure()
            fig.canvas.draw()
            data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
            clahe_hist = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            '''

            '''
            plt.subplot(4, 6, 1), plt.imshow(fixed_img)
            plt.subplot(4, 6, 2), plt.imshow(auto_img)
            plt.subplot(4, 6, 3), plt.imshow(avg_img)
            plt.subplot(4, 6, 4), plt.imshow(med_img)
            plt.subplot(4, 6, 5), plt.imshow(mode_img)
            plt.subplot(4, 6, 6), plt.imshow(clahe_img)

            plt.subplot(4, 6, 7), plt.imshow(fixed_sobel)
            plt.subplot(4, 6, 8), plt.imshow(auto_sobel)
            plt.subplot(4, 6, 9), plt.imshow(avg_sobel)
            plt.subplot(4, 6, 10), plt.imshow(med_sobel)
            plt.subplot(4, 6, 11), plt.imshow(mode_sobel)
            plt.subplot(4, 6, 12), plt.imshow(clahe_sobel)

            plt.subplot(4, 6, 13), plt.imshow(fixed_corners)
            plt.subplot(4, 6, 14), plt.imshow(auto_corners)
            plt.subplot(4, 6, 15), plt.imshow(avg_corners)
            plt.subplot(4, 6, 16), plt.imshow(med_corners)
            plt.subplot(4, 6, 17), plt.imshow(mode_corners)
            plt.subplot(4, 6, 18), plt.imshow(clahe_corners)

            plt.subplot(4, 6, 19), plt.plot(fixed_hist)
            plt.xlim([0, 255])
            plt.subplot(4, 6, 20), plt.plot(auto_hist)
            plt.xlim([0, 255])
            plt.subplot(4, 6, 21), plt.plot(avg_hist)
            plt.xlim([0, 255])
            plt.subplot(4, 6, 22), plt.plot(med_hist)
            plt.xlim([0, 255])
            plt.subplot(4, 6, 23), plt.plot(mode_hist)
            plt.xlim([0, 255])
            plt.subplot(4, 6, 24), plt.plot(clahe_hist)
            plt.xlim([0, 255])
        
            '''

            processed_images = numpy.hstack((fixed_img,
                                             auto_img,
                                             avg_img,
                                             med_img,
                                             mode_img,
                                             clahe_img))
            sobel_images = numpy.hstack((fixed_sobel,
                                         auto_sobel,
                                         avg_sobel,
                                         med_sobel,
                                         mode_sobel,
                                         clahe_sobel))
            orb_images = numpy.hstack((fixed_orb,
                                          auto_orb,
                                          avg_orb,
                                          med_orb,
                                          mode_orb,
                                          clahe_orb))
            sift_images = numpy.hstack((fixed_sift,
                                       auto_sift,
                                       avg_sift,
                                       med_sift,
                                       mode_sift,
                                       clahe_sift))
            harris_images = numpy.hstack((fixed_harris,
                                       auto_harris,
                                       avg_harris,
                                       med_harris,
                                       mode_harris,
                                       clahe_harris))
            histogram_images = numpy.hstack((fixed_hist,
                                             auto_hist,
                                             avg_hist,
                                             med_hist,
                                             mode_hist,
                                             clahe_hist))

            all = numpy.vstack((processed_images,
                                orb_images,
                                sift_images,
                                harris_images
                                ))
            rgb = cv2.cvtColor(all, cv2.COLOR_GRAY2BGR)
            video.write(rgb)



            # resize image
            width = int(all.shape[1] * 0.9)
            height = int(all.shape[0] * 0.9)
            all_resize = cv2.resize(all, (width, height))

            cv2.imshow("16 bit", all_resize)

            key_pressed = cv2.waitKey(1)
    print("Complete")
    video.release()


if __name__ == "__main__":
    main()
