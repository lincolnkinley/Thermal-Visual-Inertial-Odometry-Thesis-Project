import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from g2o_calculator import Process_g2o
from shapely.geometry import Polygon
from dijkstar import Graph, find_path
from multiprocessing import Pool

USE_MULTIPROCESSING = True

# Images will be plotted with transformations
PLOT_IMAGES = True
# Matches will be plotted
PLOT_MATCHES = True

# Uses OpenCV estimateAffinePartial2D instead of findHomography
USE_AFFINE = True

# g2o will be recalculated even if it already exists
RECALCULATE = True

# Use ORB instead of SIFT. ORB Doesn't work well so keep this False
USE_ORB = False

# Program will use the initial solution to determine overlapping images that need to be matched. Result will be found
# much faster and typically the answer will be better due to outlier removal, but recovery from a bad initial solution
# is impossible
USE_SHORTCUT = True

# Directory of input images. The output g2o will be written here
DIRECTORY = "/home/lincoln/lab/catkin_ws/src/drone_processor/src/lepton_day_complete/"

# Minimum number of matches required to consider two images a match. Must be >= 4
# This does not apply to initial solution.
MIN_MATCHES = 10

# Images will be added instead of overlaid
USE_ADD_WEIGHTED = False

# Minimum overlap with the initial solution that will result in calculating the matches
MIN_OVERLAP = 0.1

ERF_DIV = 60.0
LIN_DIV = 30.0

class Vertex:
    id = 0
    x = 0
    y = 0
    theta = 0

    def __init__(self, id, x, y, theta):
        self.id = id
        self.x = x
        self.y = y
        self.theta = theta

class Edge:
    id_a = 0
    id_b = 0
    x = 0
    y = 0
    theta = 0
    information = (0, 0, 0, 0, 0, 0)

    def __init__(self, id_a, id_b, x, y, theta, information):
        self.id_a = id_a
        self.id_b = id_b
        self.x = x
        self.y = y
        self.theta = theta
        self.information = tuple(information)


def InfoMtrx(dx, dy, dth, matches):

    covar_multiplier = 1/math.erf((matches-3)/ERF_DIV)

    exx = abs(dx/LIN_DIV) * covar_multiplier
    eyy = abs(dy/LIN_DIV) * covar_multiplier
    ett = abs(dth/(LIN_DIV * 10.0)) * covar_multiplier
    exy = 0 * covar_multiplier
    ext = 0 * covar_multiplier
    eyt = 0 * covar_multiplier

    covariance = [[exx, exy, ext],
                  [eyt, eyy, eyt],
                  [ext, exy, ett]]

    I = np.linalg.inv(covariance)
    I = [I[0, 0], I[0, 1], I[0, 2], I[1, 1], I[1, 2], I[2, 2]]
    return I


def cost_func(u, v, edge, prev_edge):
    return edge[0] * edge[1]
    return (edge[0]**2 + edge[1]**2)**0.5


def OverlayImage(top_image, bottom_image, top_x_shift = 0, top_y_shift = 0, bottom_x_shift = 0, bottom_y_shift = 0):
    top_y, top_x = top_image.shape[:2]
    top_total_x = top_x + top_x_shift
    top_total_y = top_y + top_y_shift

    bottom_y, bottom_x = bottom_image.shape[:2]
    bottom_total_x = bottom_x + bottom_x_shift
    bottom_total_y = bottom_y + bottom_y_shift

    min_x = min(top_x_shift, bottom_x_shift)
    min_y = min(top_y_shift, bottom_y_shift)

    max_x = max(top_total_x, bottom_total_x)
    max_y = max(top_total_y, bottom_total_y)

    output_x = max_x - min_x
    output_y = max_y - min_y

    background = np.zeros((output_y, output_x), dtype="uint8")
    foreground = np.zeros((output_y, output_x), dtype="uint8")

    abs_top_x_shift = abs(min_x) + top_x_shift
    abs_top_y_shift = abs(min_y) + top_y_shift

    abs_top_x = abs_top_x_shift + top_x
    abs_top_y = abs_top_y_shift + top_y

    abs_bottom_x_shift = abs(min_x) + bottom_x_shift
    abs_bottom_y_shift = abs(min_y) + bottom_y_shift

    abs_bottom_x = abs_bottom_x_shift + bottom_x
    abs_bottom_y = abs_bottom_y_shift + bottom_y

    background[abs_bottom_y_shift:abs_bottom_y, abs_bottom_x_shift:abs_bottom_x] = bottom_image[:, :]
    foreground[abs_top_y_shift:abs_top_y, abs_top_x_shift:abs_top_x] = top_image[:, :]

    if(USE_ADD_WEIGHTED):
        overlay = cv2.addWeighted(foreground, 0.25, background, 1.0, 0.0)
        return overlay

    mask = foreground > 40

    foreground = foreground*mask
    background = background*(np.logical_not(mask))

    return foreground + background


def FindMatches(image_a, kp_a, des_a, image_b, kp_b, des_b, plot_matches = False):
    index_params = dict(algorithm=1, trees=2)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    if(des_a is None or des_a.shape[0] <= 2 or des_b is None or des_b.shape[0] <= 2):
        # need at least two for Lowe's ratio test.
        return np.array([]), np.array([])
    matches = flann.knnMatch(np.float32(des_a), np.float32(des_b), k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i,(m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            matchesMask[i] = [1, 0]

    src_pts = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    if(plot_matches == True):
        img = cv2.drawMatchesKnn(image_a, kp_a, image_b, kp_b, matches, None, (0,255,0), (255,0,0), matchesMask)
        RS_Show(img, "matches")
    return src_pts, dst_pts


def DetectFeatures(detector, image):
    image_mask = (image < 1).astype(np.uint8) * 255
    blur_mask = cv2.blur(image_mask, (21, 21))
    blur_mask = (blur_mask < 1).astype(np.uint8)
    kp, des = detector.detectAndCompute(image, blur_mask)
    return (kp, des)


def FindFeatures(image_list):
    # Both Parameters
    nfeatures = 1000

    # SIFT Parameters, Original parameters
    nOctaveLayers = 3
    contrastThreshold = 0.04
    SIFT_edgeThreshold = 10
    sigma = 1.6

    # SIFT Parameters, Optimized parameters, These are dialed in so well that we get a nearly perfect initial solution.
    # Don't use these so we can show GTSAM improves the solution
    nOctaveLayers = 5
    contrastThreshold = 0.03
    SIFT_edgeThreshold = 7
    sigma = 0.7

    # Optimization Blackfly
    nOctaveLayers = 9
    contrastThreshold = 0.03
    SIFT_edgeThreshold = 16
    sigma = .7

    # Optimization Blackfly Night
    nOctaveLayers = 9
    contrastThreshold = 0.01
    SIFT_edgeThreshold = 16
    sigma = .7


    # Optimization Boson
    nOctaveLayers = 9
    contrastThreshold = 0.03
    SIFT_edgeThreshold = 5
    sigma = 0.7

    # Optimization Lepton
    nOctaveLayers = 9
    contrastThreshold = 0.03
    SIFT_edgeThreshold = 16
    sigma = .7

    # ORB Parameters, Default Parameters
    scaleFactor = 1.1
    nlevels = 18
    ORB_edgeThreshold = 16
    firstLevel = 0
    WTA_K = 3
    scoreType = cv2.ORB_HARRIS_SCORE
    patchSize = 16
    fastThreshold = 10

    if (USE_ORB):
        detector = cv2.ORB_create(nfeatures, scaleFactor, nlevels, ORB_edgeThreshold, firstLevel, WTA_K, scoreType,
                                  patchSize, fastThreshold)
    else:
        detector = cv2.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, SIFT_edgeThreshold, sigma)

    border = 1

    keypoints = []
    descriptors = []

    i = 0
    size = len(image_list)

    for image in image_list:
        kp, des = DetectFeatures(detector, image)
        keypoints.append(kp)
        descriptors.append(des)
        print(f"Detected Features {i} out of {size}")
        i += 1
    return keypoints, descriptors


def CalculateHomography(src_pts, dst_pts, min_matches):
    H = np.eye(3)
    num_inliers = 0
    if (src_pts.shape[0] >= 4 and dst_pts.shape[0] >= 4):
        if (USE_AFFINE == True):
            af, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, None, cv2.RANSAC, 5.0)
            if(af is None):
                H = np.eye(3)
                num_inliers = 0
            else:
                H = np.vstack((af, [[0, 0, 1]]))
        else:
            H, inliers = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 1.4)
            if(H is None):
                H = np.eye(3)
                num_inliers = 0
        num_inliers = np.count_nonzero(inliers)
        if (num_inliers < min_matches):
            # There were enough matches to qualify, but after RANSAC the matches dropped below the minimum
            H = np.eye(3)
            num_inliers = 0
    return H, num_inliers


def ImageOverlap(image_a, transform_a, image_b, transform_b):
    height_a, width_a = image_a.shape[:2]
    corners_a = np.float32([[0, 0], [width_a, 0], [width_a, height_a], [0, height_a]]).reshape(-1, 1, 2)
    projected_corners_a = cv2.perspectiveTransform(corners_a, transform_a).reshape(4, 2)

    height_b, width_b = image_b.shape[:2]
    corners_b = np.float32([[0, 0], [width_b, 0], [width_b, height_b], [0, height_b]]).reshape(-1, 1, 2)
    projected_corners_b = cv2.perspectiveTransform(corners_b, transform_b).reshape(4, 2)

    polygon_a = Polygon(projected_corners_a)
    polygon_b = Polygon(projected_corners_b)

    try:
        intersection_area = polygon_a.intersection(polygon_b).area
    except:
        print("Error calculating polygon")
        return 1.0
    largest_polygon_area = polygon_a.area if polygon_a.area > polygon_b.area else polygon_b.area

    return intersection_area/largest_polygon_area


def CalculateInitialSolution(image_list, keypoints_list, descriptors_list):
    if(len(image_list) != len(keypoints_list) or len(image_list) != len(descriptors_list)):
        print(f"ERROR: Invalid input lengths, should all be equal.\n  image_list: {len(image_list)}\n  keypoints_list: {len(keypoints_list)}\n  descriptors_list: {len(descriptors_list)}")

    homographies = np.zeros((len(image_list), len(image_list), 3, 3), dtype="float")
    num_matches = np.zeros((len(image_list), len(image_list)), dtype="int")
    for i in range(len(image_list) - 1):
        if(i == 443):
            print("f")
            pass
        src_pts, dst_pts = FindMatches(image_list[i], keypoints_list[i], descriptors_list[i], image_list[i+1], keypoints_list[i+1], descriptors_list[i+1], PLOT_MATCHES)
        H, num_inliers = CalculateHomography(src_pts, dst_pts, 4)
        print(f"Initial Solution: Found {num_inliers} matches between images {i} and {i+1}")
        if(num_inliers < 20):
            print("Poor Initial Solution, trying with lots more features")

            nfeatures = 8000
            nOctaveLayers = 9
            contrastThreshold = 0.03
            SIFT_edgeThreshold = 16
            sigma = .7
            detector = cv2.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, SIFT_edgeThreshold, sigma)
            kp_a, des_a = DetectFeatures(detector, image_list[i])
            kp_b, des_b = DetectFeatures(detector, image_list[i+1])
            src_pts, dst_pts = FindMatches(image_list[i], kp_a, des_a, image_list[i+1], kp_b, des_b, PLOT_MATCHES)
            H, num_inliers = CalculateHomography(src_pts, dst_pts, 4)
        homographies[i, i+1] = H
        num_matches[i, i+1] = num_inliers
    return homographies, num_matches


def Pool_Func(data):
    i, image_list, keypoints_list, descriptors_list, image_transforms = data
    homographies = np.zeros((1, len(image_list), 3, 3), dtype="float")
    num_matches = np.zeros((1, len(image_list)), dtype="int")



    for j in range(i + 2, len(image_list)):
        overlap_percent = ImageOverlap(image_list[i], image_transforms[i], image_list[j], image_transforms[j])

        if ((overlap_percent < MIN_OVERLAP) and (USE_SHORTCUT == True)):
            continue
        src_pts, dst_pts = FindMatches(image_list[i], keypoints_list[i], descriptors_list[i], image_list[j],
                                       keypoints_list[j], descriptors_list[j])
        H, num_inliers = CalculateHomography(src_pts, dst_pts, MIN_MATCHES)
        print(f"Optimized Solution: Found {num_inliers} matches between images {i} and {j}")
        homographies[0, j] = H
        num_matches[0, j] = num_inliers
    return (homographies, num_matches)

def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle, point.response, point.octave, point.class_id)

import copyreg
copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

def OptimizeSolution(image_list, keypoints_list, descriptors_list, initial_solution, initial_num_matches):
    homographies = initial_solution
    num_matches = initial_num_matches
    image_transforms = np.zeros((len(image_list), 3, 3), dtype="float")
    transform = np.eye(3)
    image_transforms[0] = transform
    for i in range(len(image_list) - 1):
        transform = np.matmul(transform, initial_solution[i, i+1])
        image_transforms[i+1] = transform
    if (USE_MULTIPROCESSING):
        data = []
        for i in range(len(image_list)):
            data.append((i, image_list, keypoints_list, descriptors_list, image_transforms))

        p = Pool(4)
        out_data = p.map(Pool_Func, data)
        for i in range(len(out_data) - 2):
            new_homographies = out_data[i][0]
            new_num_matches = out_data[i][1]

            num_matches[i, i + 2:] = new_num_matches[0, i + 2:]
            homographies[i,i+2:] = new_homographies[0, i+2:]

        return homographies, num_matches

    for i in range(len(image_list)):
        for j in range(i+2, len(image_list)):
            overlap_percent = ImageOverlap(image_list[i], image_transforms[i], image_list[j], image_transforms[j])
            if(overlap_percent < MIN_OVERLAP and (USE_SHORTCUT == True)):
                continue
            src_pts, dst_pts = FindMatches(image_list[i], keypoints_list[i], descriptors_list[i], image_list[j], keypoints_list[j], descriptors_list[j], PLOT_MATCHES)
            H, num_inliers = CalculateHomography(src_pts, dst_pts, MIN_MATCHES)
            print(f"Optimized Solution: Found {num_inliers} matches between images {i} and {j}")
            homographies[i, j] = H
            num_matches[i, j] = num_inliers
    return homographies, num_matches


def ProcessImage(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def Parse_g2o(filename):
    vertices = []
    edges = []
    with open(filename, 'r') as g2o_file:
        lines = g2o_file.readlines()
        for line in lines:
            words = line.split()
            if (words[0] == "EDGE_SE2"):
                edge = Edge(int(words[1]),
                            int(words[2]),
                            float(words[3]),
                            float(words[4]),
                            float(words[5]),
                            (float(words[6]),
                             float(words[7]),
                             float(words[8]),
                             float(words[9]),
                             float(words[10]),
                             float(words[11])))
                edges.append(edge)
            if (words[0] == "VERTEX_SE2"):
                vertex = Vertex(int(words[1]),
                                float(words[2]),
                                float(words[3]),
                                float(words[4]))
                vertices.append(vertex)
    return vertices, edges


def get_corrected_rotation_matrix(H):
    R = H[0:2,0:2]
    U,S,V = np.linalg.svd(R)
    R_corr = U@V
    return R_corr


def Create_g2o(homographies, filename, num_matches):
    vertices = [f"VERTEX_SE2 0 0.0 0.0 0.0\n"]
    current_transform = np.eye(3)
    for i in range(homographies.shape[0] - 1):
        current_transform = np.matmul(current_transform, homographies[i, i + 1])
        dx = current_transform[0, 2]
        dy = current_transform[1, 2]
        corrected_rotation = get_corrected_rotation_matrix(current_transform)
        uncorrected = math.atan2(current_transform[1, 0], current_transform[0, 0])
        dth = math.atan2(corrected_rotation[1, 0], corrected_rotation[0, 0])
        if (round(uncorrected, 8) != round(dth, 8)):
            print(f"Error: {uncorrected}, {dth}")

        g2o_string = f"VERTEX_SE2 {i + 1} {dx} {dy} {dth}\n"
        vertices.append(g2o_string)

    edges = []
    for i in range(homographies.shape[0]):
        for j in range(i, homographies.shape[1]):
            H = homographies[i,j]
            matches = num_matches[i,j]
            if(np.array_equal(H, np.eye(3)) or np.array_equal(H, np.zeros((3,3), dtype="float"))):
                # Either image is identical, or no matches found
                print(f"No edges between {i} and {j}")
            else:
                print(f"Found edge between {i} and {j}")
                x = H[0, 2]
                y = H[1, 2]
                corrected_rotation = get_corrected_rotation_matrix(H)
                th = math.atan2(corrected_rotation[1, 0], corrected_rotation[0, 0])
                uncorrected = math.atan2(H[1, 0], H[0, 0])
                if(round(uncorrected, 8) != round(th, 8)):
                    print(f"Error: {uncorrected}, {th}")
                #th = 1 * np.pi / 180
                I = InfoMtrx(x, y, th, matches)
                g2o_string = f"EDGE_SE2 {i} {j} {x} {y} {th} {I[0]} {I[1]} {I[2]} {I[3]} {I[4]} {I[5]}\n"
                edges.append(g2o_string)

    with open(filename, 'w') as g2o_file:
        for vertex_string in vertices:
            g2o_file.write(vertex_string)
        for edge_string in edges:
            g2o_file.write(edge_string)


def Plot_g2o_Unoptimized(filename, color):
    vertices, edges = Parse_g2o(filename)

    sequential_edges = []

    for i in range(len(vertices) - 1):
        found_flag = False
        for edge in edges:
            if(i == edge.id_a and edge.id_b == edge.id_a + 1):
                sequential_edges.append(edge)
                found_flag = True
        if(found_flag == False):
            sequential_edges.append(Edge(i, i + 1, 0, 0, 0, [float("inf"), 0,0, float("inf"), 0, float("inf")]))

    prev_vertex = None
    for vertex in vertices:

        covar_x = 0
        covar_y = 0

        for i in range(vertex.id):
            edge = sequential_edges[i]
            info = [[edge.information[0], 0, 0], [0, edge.information[3], 0], [0, 0, edge.information[5]]]
            covariance = np.linalg.inv(info)
            covar_x += covariance[0, 0]
            covar_y += covariance[1, 1]

        th = np.linspace(0, 2 * np.pi, 100)
        x = np.abs(covar_x) * np.cos(th)
        y = np.abs(covar_y) * np.sin(th)
        plt.figure(200)
        plt.plot(vertex.x + x, vertex.y + y, color)
        if(not(prev_vertex is None)):
            plt.plot([vertex.x, prev_vertex.x], [vertex.y, prev_vertex.y], color)

        prev_vertex = vertex


def Plot_g2o_Optimized(filename, color):
    vertices, edges = Parse_g2o(filename)
    graph = Graph()

    for edge in edges:
        info = [[edge.information[0], 0, 0], [0, edge.information[3], 0], [0, 0, edge.information[5]]]
        covariance = np.linalg.inv(info)

        graph.add_edge(edge.id_a, edge.id_b, (covariance[0, 0], covariance[1, 1]))
        graph.add_edge(edge.id_b, edge.id_a, (covariance[0, 0], covariance[1, 1]))

    prev_vertex = None
    for vertex in vertices:

        covar_x = 0
        covar_y = 0

        if (vertex.id != 0):
            path = None
            attempts = 0
            while(path is None):
                # find_path will fail to find a path to and from the same node, so skip the first vertex
                try:
                    path = find_path(graph, 0, vertex.id-attempts, cost_func=cost_func)
                    for edge in path.edges:
                        covar_x += edge[0]
                        covar_y += edge[1]
                except:
                    print(f"Find fath to {vertex.id} failed. Trying {vertex.id-(attempts+1)}")
                attempts+=1

        th = np.linspace(0, 2 * np.pi, 100)
        x = np.abs(covar_x) * np.cos(th)
        y = np.abs(covar_y) * np.sin(th)
        plt.figure(200)
        plt.plot(vertex.x + x, vertex.y + y, color)
        if(not(prev_vertex is None)):
            plt.plot([vertex.x, prev_vertex.x], [vertex.y, prev_vertex.y], color)

        prev_vertex = vertex


def RS_Show(image, name="image"):
    if(image.shape[0] > 1080):
        scale = 1080 / image.shape[0]
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    if(image.shape[1] > 1920):
        scale = 1920 / image.shape[1]
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow(name, image)
    k = cv2.waitKey(1)


def Mosaic(images, transforms, name):
    accumulator_image = np.zeros((1,1))

    sum_x_shift = 0
    sum_y_shift = 0

    loop = 0
    pct = 0
    size = len(images)

    for i in range(len(images)):
        add_image = images[i]

        transform = transforms[i]

        # Taken from https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
        h, w = add_image.shape[:2]
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
        warped_image = cv2.warpPerspective(add_image, transform, (math.ceil(xmax - xmin), math.ceil(ymax - ymin)))

        accumulator_image = OverlayImage(warped_image, accumulator_image, xmin - sum_x_shift, ymin - sum_y_shift)
        if (xmin < sum_x_shift):
            sum_x_shift = xmin
        if (ymin < sum_y_shift):
            sum_y_shift = ymin
        RS_Show(accumulator_image, "Optimized")
        if ((100 * float(loop)) / float(size) > float(pct)):
            print(str(pct) + "% complete")
            pct += 1

        loop += 1

    return accumulator_image


def VisualOdometry(directory):
    images = []

    # Max, for debugging, set to zero for off
    max = 0
    start = 0
    loops = 0

    print("Loading images...")
    for filename in sorted(os.listdir(directory)):
        if (filename.endswith(".png")):
            loops += 1
            if(RECALCULATE == True and loops < start):
                continue
            image = cv2.imread(directory + filename)
            image = ProcessImage(image)
            images.append(image)

            if(max > 0 and len(images) >= max):
                break

    if((not os.path.isfile(directory + "output.g2o")) or RECALCULATE == True):
        # g2o does not exist, need to create it
        if (len(os.listdir(directory)) < 2):
            print("There must be at least two images to stitch together!")
            return

        print("Finding Features")
        keypoints, descriptors = FindFeatures(images)

        print("Initial Solution")
        initial_homographies, initial_num_matches = CalculateInitialSolution(images, keypoints, descriptors)
        print("Optimizing Solution")
        optimized_homographies, optimized_num_matches = OptimizeSolution(images, keypoints, descriptors, initial_homographies, initial_num_matches)
        print("Creating g2o")
        Create_g2o(optimized_homographies, directory + "output.g2o", optimized_num_matches)
    print("Processing g2o")
    #Process_g2o(directory + "output.g2o", directory + "optimized.g2o")
    #print("Plotting output")
    #Plot_g2o_Unoptimized(directory + "output.g2o", 'r')
    #print("Plotting Optimized")
    #Plot_g2o_Optimized(directory + "optimized.g2o", 'g')
    #plt.title(f"Visual Odometry | ERF_DIV: {ERF_DIV}, LIN_DIV: {LIN_DIV}")
    #plt.show()

    if(PLOT_IMAGES == True):
        print("Begin Mosiacing")
        output_vertices, _ = Parse_g2o(directory + "output.g2o")
        #optimized_vertices, _ = Parse_g2o(directory + "optimized.g2o")

        output_transforms = []
        #optimized_transforms = []

        for vertex in output_vertices:
            transform = np.zeros((3,3))

            transform[0, 0] = math.cos(vertex.theta)
            transform[0, 1] = -math.sin(vertex.theta)
            transform[1, 0] = math.sin(vertex.theta)
            transform[1, 1] = math.cos(vertex.theta)

            transform[0, 2] = vertex.x
            transform[1, 2] = vertex.y
            transform[2, 2] = 1

            output_transforms.append(transform)
        '''
        for vertex in optimized_vertices:
            transform = np.zeros((3, 3))

            transform[0, 0] = math.cos(vertex.theta)
            transform[0, 1] = -math.sin(vertex.theta)
            transform[1, 0] = math.sin(vertex.theta)
            transform[1, 1] = math.cos(vertex.theta)

            transform[0, 2] = vertex.x
            transform[1, 2] = vertex.y
            transform[2, 2] = 1

            optimized_transforms.append(transform)
        '''

        print("Calculating Optimized Mosaic")
        opimized_mosiac = Mosaic(images, output_transforms, "Unoptimized")
        RS_Show(opimized_mosiac, "Unoptimized")

        cv2.imwrite(DIRECTORY + "Unptimized.png", opimized_mosiac)
    #plt.title(f"Visual Odometry | ERF_DIV: {ERF_DIV}, LIN_DIV: {LIN_DIV}")
    #plt.show()
    cv2.waitKey(0)

def main():
    VisualOdometry(DIRECTORY)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
