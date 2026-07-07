import numpy as np
import line_profiler

@line_profiler.profile
def describeKeypoints(img, keypoints, r):
    """
    Returns a (2r+1)^2xN matrix of image patch vectors based on image img and a 2xN matrix containing the keypoint
    coordinates. r is the patch "radius".
    """
    len = keypoints.shape[1]
    descriptors = np.zeros((len, 2*r+1, 2*r+1))
    img_padded = np.pad(img, r+1)
    
    for i in range(len):
        x = keypoints[1, i] + r+1
        y = keypoints[0, i] + r+1
        descriptors[i] = img_padded[y-r:y+r+1, x-r:x+r+1]
        
    descriptors = descriptors.reshape((len, (2*r+1)**2)).T
    return descriptors
