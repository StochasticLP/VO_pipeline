import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

SCALES_PER_OCTAVE = 3
BASE_SIGMA = 1.6
def computeImagePyramid(img, num_octaves):
    h, w = img.shape
    pyramid = []
    for i in range(num_octaves):
        
    num_scales = SCALES_PER_OCTAVE+2


    #create difference of gaussian pyramid
    dog_pyramid = {0: np.zeros((num_scales, h, w))}
    base_image = img
    for octave in range(num_octaves):
        if (octave > 0):
            base_image = dog_pyramid[octave-1][-2, ::2,::2]
            dog_pyramid[octave] = np.zeros((num_scales, h/2**octave, w/2**octave))
            
        for s in range(num_scales+1):
            #apply gaussian filter to images in octave
            blurred_images = np.zeros((num_scales+1, h/2**octave, w/2**octave))
            sigma = 2**((s-1)/SCALES_PER_OCTAVE)*BASE_SIGMA
            blurred_images[s] = gaussian_filter(base_image, sigma)
            
            #take difference of each layer
            if (s > 0):
                dog_pyramid[octave][s-1] = blurred_images[s] - blurred_images[s-1]
            
            
    return dog_pyramid
