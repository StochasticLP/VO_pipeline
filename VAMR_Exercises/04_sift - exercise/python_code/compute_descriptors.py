import cv2
import numpy as np

def getGaussianKernel(size, sigma):
    k1d = cv2.getGaussianKernel(size, sigma)   # shape (16, 1)
    K = k1d @ k1d.T                                # shape (16, 16)


def getImageGradient(image):
    img = image.astype(np.float64)
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    norms, angles = cv2.cartToPolar(Ix, Iy, angleInDegrees=False)
    return norms, angles


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
    gaussianKernel = getGaussianKernel(16, 1.5)
    keypoint_descriptors = []

    
    for i in range(len(keypoint_locations)):
        keypoints = keypoint_locations[i]
        
        octave_descriptors = []
        #get 16x16 patch around keypoint from blurred image with index (scale-1) from its octave
        for j in range(len(keypoints)):
            keypoint = keypoints[j]
            s = keypoint[0]
            u = keypoint[1]
            v = keypoint[2]
            patch = blurred_images[i][s-1, (u-8):(u+7), (v-8):(v+7)]
            
            #get norm and orientation of gradients of patch
            norms, angles = getImageGradient(patch)
            
            #scale norms with sigma_w = 1.5 * 16 gaussian kernel
            norms = norms*gaussianKernel
            
            keypoint_descriptor = []
            for k in range(0, 16, 4):
                for l in range(0, 16, 4):
                    descriptor_angles = angles[k:(k+4),(l):(l+4)].flatten
                    descriptor_norms = norms[k:(k+4),(l):(l+4)].flatten
                    keypoint_descriptor.append(np.histogram(descriptor_angles, bins=8, weights=descriptor_norms))
            octave_descriptors.append(keypoint_descriptor)
            
        keypoint_descriptors.append(octave_descriptors)
    
    # return descriptors and final keypoint locations
    return keypoint_descriptors, keypoint_locations
