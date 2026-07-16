import cv2
import numpy as np
import matplotlib.pyplot as plt

from compute_blurred_images import computeBlurredImages
from compute_descriptors import computeDescriptors 
from compute_difference_of_gaussians import computeDifferenceOfGaussians 
from compute_image_pyramid import computeImagePyramid 
from extract_keypoints import extractKeypoints

def main(rotation_invariant, rotation_img2_deg, contrast_threshold, sift_sigma, 
        rescale_factor, num_scales, num_octaves):

    # Convenience function to read in images into grayscale and convert them to double 
    get_image = lambda fname, scale: \
            cv2.normalize(
                cv2.resize( \
                    cv2.imread(
                        fname, cv2.IMREAD_GRAYSCALE), (0,0), fx = scale, fy = scale
                    ).astype('float'), \
                None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    # Read in images
    img1 = get_image("../data/img_1.jpg", rescale_factor)
    img2 = get_image("../data/img_2.jpg", rescale_factor)
    

    # If we want to test our rotation invariant features, rotate the second image
    if np.abs(rotation_img2_deg) > 1e-6:
        pass
        # Lets go and rotate the image
        # - get the original height and width
        # - create rotation matrix
        # - calculate the size of the rotated image
        # - pad the image
        # - rotate the image
    # TODO: Your code here
    

    # Actually compute the SIFT features. For both images do:
    # - construct the image pyramid
    # - compute the blurred images
    # - compute difference of gaussians
    # - extract the keypoints
    # - compute the descriptors
    imgs = [img1, img2]
    keypoint_locations = []
    keypoint_descriptors = []

    for i in range(len(imgs)):
        pass
    # TODO: Your code here
    
    # OpenCV brute force matching
    """ Remove this comment if you have completed the code until here
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(keypoint_descriptors[0].astype(np.float32), keypoint_descriptors[1].astype(np.float32), 2)
    """

    # Apply ratio test
    # TODO: Your code here

    # Plot the results
    """ Remove this comment if you have completed the code until here
    plt.figure()
    dh = int(img2.shape[0] - img1.shape[0])
    top_padding = int(dh/2)
    img1_padded = cv2.copyMakeBorder(img1, top_padding, dh - int(dh/2),
            0, 0, cv2.BORDER_CONSTANT, 0)
    plt.imshow(np.c_[img1_padded, img2], cmap = "gray")

    for match in good:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        x1 = keypoint_locations[0][img1_idx,1]
        y1 = keypoint_locations[0][img1_idx,0] + top_padding
        x2 = keypoint_locations[1][img2_idx,1] + img1.shape[1]
        y2 = keypoint_locations[1][img2_idx,0]
        plt.plot(np.array([x1, x2]), np.array([y1, y2]), "o-")
    plt.show()
    """


if __name__=="__main__":
    # User parameters
    rotation_invariant =True       # Enable rotation invariant SIFT
    rotation_img2_deg = 60          # Rotate the second image to be matched

    # sift parameters
    contrast_threshold = 0.04       # for feature matching
    sift_sigma = 1.0                # sigma used for blurring
    rescale_factor = 0.3            # rescale images to make it faster
    num_scales = 3                  # number of scales per octave
    num_octaves = 5                 # number of octaves
        
    main(rotation_invariant, rotation_img2_deg, contrast_threshold, sift_sigma, 
            rescale_factor, num_scales, num_octaves)
