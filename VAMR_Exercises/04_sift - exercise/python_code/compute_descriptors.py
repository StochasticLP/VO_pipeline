import cv2
import numpy as np

def getGaussianKernel(size, sigma):
    pass
    # TODO: Your code here


def getImageGradient(image):
    pass
    # TODO: Your code here


def derotatePatch(img, loc, patch_size, orientation):
    # it can't be worse than a 45 degree rotation, so lets pad 
    # under this assumption. Then it will be enough for sure.
    pass
    # TODO: Your code here
    
    # compute derotated patch  
    for px in range(patch_size):
        for py in range(patch_size):
            pass
            
    # TODO: Your code here

            # rotate patch by angle ori
    # TODO: Your code here

            # move coordinates to patch
    # TODO: Your code here

            # sample image (using nearest neighbor sampling as opposed to more
            # accuracte bilinear sampling)
    # TODO: Your code here
    # Return the patch
    # TODO: Your code here


def computeDescriptors(blurred_images, keypoint_locations, rotation_invariant):
    # return descriptors and final keypoint locations
    pass
    # TODO: Your code here







