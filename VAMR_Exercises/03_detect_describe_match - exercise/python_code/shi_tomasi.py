import numpy as np
from scipy import signal


def shi_tomasi(img, patch_size):
    """ Returns the shi-tomasi scores for an image and patch size patch_size
        The returned scores are of the same shape as the input image """

    #define sobel filter
    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    sobel_y = np.array ([[-1, -2, -1],
                     [0, 0, 0,],
                     [1, 2, 1]])

    #convolve sobel filter with image to produce I_x and I_y matrices
    Ix = signal.convolve2d(img, sobel_x, mode='valid')
    Iy = signal.convolve2d(img, sobel_y, mode = 'valid')

    #compute I_x^2, I_y^2, I_xI_y
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    #compute sum I_x^2, sum I_y^2, sum I_xI_y
    #convolve patch size matrix with intenisty matrices
        #compute 3 M coeficceints
    patch = np.ones((patch_size, patch_size))
    sum_Ixx = signal.convolve2d(patch, Ixx, mode='valid')
    sum_Iyy = signal.convolve2d(patch, Iyy, mode='valid')
    sum_Ixy = signal.convolve2d(patch, Ixy, mode='valid')

    #Compute eigenvalues using quadratic formula
    disc = (sum_Ixx - sum_Iyy)**2 + 4 * sum_Ixy**2
    score = ((sum_Ixx + sum_Iyy) - np.sqrt(np.maximum(disc, 0))) / 2
    pad = 1 + patch_size // 2  # accounts for Sobel shrink + patch shrink
    score = np.pad(score, pad)
    score[score < 0] = 0

    #return min eigenvalue for each pixel
    return score
