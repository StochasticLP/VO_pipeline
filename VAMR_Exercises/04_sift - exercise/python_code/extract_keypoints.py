import cv2
import numpy as np
from scipy.ndimage import maximum_filter

def extractKeypoints(diff_of_gaussians, contrast_threshold):
    # returns the keypoint locations
    
    #non maximum suppression
    diff_of_gaussians = np.array(diff_of_gaussians)
    diff_of_gaussians[diff_of_gaussians < contrast_threshold] = 0
    
    keypoints = []
    for i in range(len(diff_of_gaussians)):
        #apply maximum filter
        dog_volume = diff_of_gaussians[i]
        maximums = maximum_filter(dog_volume, size=3)
        maximums[[0, -1]] = -1
        keypoints[i] = np.argwhere(dog_volume == maximums)
    
    return keypoints
