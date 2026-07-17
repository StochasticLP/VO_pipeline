import cv2

def computeImagePyramid(img, num_octaves):
    h, w = img.shape
    pyramid = []
    
    for i in range(num_octaves):
        pyramid.append(img[::2**i, ::2**i])
        
    return pyramid
