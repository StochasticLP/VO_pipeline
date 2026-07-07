import numpy as np
from scipy.ndimage import uniform_filter
import cv2

import line_profiler

#define sobel filter
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

@line_profiler.profile
def harris(img, patch_size, kappa):
    """ Returns the harris scores for an image given a patch size and a kappa value
        The returned scores are of the same shape as the input image """
        

    #convolve sobel filter with image to produce I_x and I_y matrices
    Ix = cv2.filter2D(img.astype(np.float64), -1, SOBEL_X)
    Iy = cv2.filter2D(img.astype(np.float64), -1, SOBEL_Y)

    #compute I_x^2, I_y^2, I_xI_y
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    #compute sum I_x^2, sum I_y^2, sum I_xI_y
    #uniform_filter is a separated implementation
    
    sum_Ixx = uniform_filter(Ixx, size=patch_size)
    sum_Iyy = uniform_filter(Iyy, size=patch_size)
    sum_Ixy = uniform_filter(Ixy, size=patch_size)

    #Compute eigenvalues using quadratic formula
    score = (sum_Ixx * sum_Iyy - sum_Ixy**2) - kappa * (sum_Ixx + sum_Iyy)**2

    #make matrix original size of image
    pad = 1 + patch_size // 2
    score = np.pad(score, pad)
    
    return score

