import cv2
import numpy as np

def computeDifferenceOfGaussians(blurred_images):
    # The number of octaves can be inferred from the length of blurred_images
    dog_pyramid = []
    
    for i in range(len(blurred_images)):
        num_scales, h, w = blurred_images[i].shape
        
        dog_volume = np.zeros((num_scales, h, w))
        for s in range(num_scales):
            if s > 0:
                dog_volume[s-1] = blurred_images[i][s] - blurred_images[i][s-1]
                
        dog_pyramid.append(dog_volume)
        
    return dog_pyramid


