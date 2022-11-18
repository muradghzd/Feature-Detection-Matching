import cv2
import numpy as np
import skimage

# Global variables (tuned)
alpha = 0.06
sobel_v = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
sobel_h = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
args1 = [(0,0), 0.15]
args2 = [(0, 0), 0.167]
args3 = [(0,0), 0.295]
threshold1 = 0.04
threshold2 = 0.15
threshold3 = 0.9

def get_interest_points(image, descriptor_window_image_width):
    # Local Feature Stencil Code
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of interest points for the input image

    # 'image' can be grayscale or color, your choice.
    # 'descriptor_window_image_width', in pixels.
    #   This is the local feature descriptor width. It might be useful in this function to
    #   (a) suppress boundary interest points (where a feature wouldn't fit entirely in the image, anyway), or
    #   (b) scale the image filters being used.
    # Or you can ignore it.

    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.

    # Implement the Harris corner detector (See Szeliski 4.1.1) to start with.

    # If you're finding spurious interest point detections near the boundaries,
    # it is safe to simply suppress the gradients / corners near the edges of
    # the image.

    # My implementation
    m, n = image.shape
    C = np.zeros((m, n))    
    w = descriptor_window_image_width//2
    image = cv2.GaussianBlur(image, *args1)
 
    I_x = cv2.filter2D(image, -1, kernel=sobel_v)
    I_y = cv2.filter2D(image, -1, kernel=sobel_h)

    I_xx = I_x*I_x
    I_xy = I_x*I_y
    I_yy = I_y*I_y
    
    I_xx = cv2.GaussianBlur(I_xx, *args1) 
    I_yy = cv2.GaussianBlur(I_yy, *args1)
    I_xy = cv2.GaussianBlur(I_xy, *args1)

    for i in range(w, m-w+1):
        for j in range(w, n-w+1):
            g_XX = np.sum(I_xx[i-w:i+w,j-w:j+w])
            g_YY = np.sum(I_yy[i-w:i+w,j-w:j+w])
            g_XY = np.sum(I_xy[i-w:i+w,j-w:j+w])

            c = g_XX*g_YY-g_XY*g_XY-alpha*(g_XX+g_YY)*(g_XX+g_YY)
            if c > threshold1:
                C[i,j] = c

    y, x = skimage.feature.peak_local_max(C, 3, threshold1).T

    return x, y

    # After computing interest points, here's roughly how many we return
    # For each of the three image pairs
    # - Notre Dame: ~1300 and ~1700
    # - Mount Rushmore: ~3500 and ~4500
    # - Episcopal Gaudi: ~1000 and ~9000



def get_descriptors(image, x, y, descriptor_window_image_width):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of feature descriptors for a given set of interest points.

    # 'image' can be grayscale or color, your choice.
    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
    #   The local features should be centered at x and y.
    # 'descriptor_window_image_width', in pixels, is the local feature descriptor width.
    #   You can assume that descriptor_window_image_width will be a multiple of 4
    #   (i.e., every cell of your local SIFT-like feature will have an integer width and height).
    # If you want to detect and describe features at multiple scales or
    # particular orientations, then you can add input arguments.

    # 'features' is the array of computed features. It should have the
    #   following size: [length(x) x feature dimensionality] (e.g. 128 for
    #   standard SIFT)

    # My implementation
    a, b = image.shape[:2]
    m = x.size
    w_h = descriptor_window_image_width//2
    w_q = descriptor_window_image_width//4

    features = np.zeros((m, w_q, w_q, 8))
    image = cv2.GaussianBlur(image, *args2)
    
    i_x = cv2.filter2D(image,-1,kernel=sobel_v)
    i_y = cv2.filter2D(image, -1,kernel=sobel_h)
    
    dir_grad = np.arctan2(i_y, i_x)
    dir_grad = np.where(dir_grad>=0,dir_grad,dir_grad+2*np.pi)
    mag_grad = np.sqrt(i_x**2 + i_y**2)
    
    for i in range(m):
        # Suppresing border values
        y_i = min(max(int(y[i]), w_h), a-w_h)
        x_i = min(max(int(x[i]), w_h), b-w_h) 

        mag_w = cv2.GaussianBlur(mag_grad[y_i-w_h:y_i+w_h, x_i-w_h:x_i+w_h], *args3)
        dir_w = cv2.GaussianBlur(dir_grad[y_i-w_h:y_i+w_h, x_i-w_h:x_i+w_h], *args3)

        features = do_bin(features,mag_w,dir_w,descriptor_window_image_width,i)


    features = np.reshape(features, (m, -1))
    
    features_sq = np.linalg.norm(features, axis=1)
    features = features / np.reshape(features_sq, (-1, 1))
    
    features = np.where(features<threshold2, features, threshold2)
    
    features_sq = np.linalg.norm(features, axis=1)
    features = features / np.reshape(features_sq, (-1, 1)) 

    return features

def match_features(features1, features2):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Please implement the "nearest neighbor distance ratio test",
    # Equation 4.18 in Section 4.1.3 of Szeliski.

    #
    # Please assign a confidence, else the evaluation function will not work.
    #

    # This function does not need to be symmetric (e.g., it can produce
    # different numbers of matches depending on the order of the arguments).

    # Input:
    # 'features1' and 'features2' are the n x feature dimensionality matrices.
    #
    # Output:
    # 'matches' is a k x 2 matrix, where k is the number of matches. The first
    #   column is an index in features1, the second column is an index in features2.
    #
    # 'confidences' is a k x 1 matrix with a real valued confidence for every match.

    # My implementation
    m, n = features1.shape[0], features2.shape[0]

    matches = []
    confidences = []
    for i in range(m):
        f1 = features1[i]
        diff = features2 - f1
        norm_diff = np.linalg.norm(diff, axis=1)
        nearest_two = np.argsort(norm_diff)
        ratio = norm_diff[nearest_two[0]] / norm_diff[nearest_two[1]]
        if ratio < threshold3:
            matches.append([i, nearest_two[0]])
            confidences.append(1-ratio)
    matches = np.array(matches)
    confidences = np.array(confidences)

    return matches, confidences

# Helper function(s)
def do_bin(fs, mag_w, dir_w, width,i):
    w_q = width//4
    for x in range(w_q):
        for y in range(w_q):
            mag_xy = mag_w[x*w_q: (x+1)*w_q, y*w_q:(y+1)*w_q]

            dir_xy = dir_w[x*w_q: (x+1)*w_q, y*w_q:(y+1)*w_q]

            fs[i, x, y] = np.histogram(dir_xy, bins=8, range=(0, 2*np.pi), weights=mag_xy)[0]

    return fs
