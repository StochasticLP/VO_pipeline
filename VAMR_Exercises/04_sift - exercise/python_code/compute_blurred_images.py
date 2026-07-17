import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def computeBlurredImages(image_pyramid, num_scales, sift_sigma):
    # The number of octaves can be inferred from the length of the image pyramid
    blurred_volumes = []
    
    for i in range(len(image_pyramid)):
        image = image_pyramid[i]
        h, w = image.shape
        
        blurred_images = np.zeros((num_scales + 3, h, w))
        for s in range(num_scales + 3):
            #apply gaussian filter to images in octave
            #Start with s = -1, S = num_scales
            sigma = 2**((s-1)/(num_scales))*sift_sigma
            blurred_images[s] = gaussian_filter(image, sigma)
            
        blurred_volumes.append(blurred_images)
        
    return blurred_volumes
