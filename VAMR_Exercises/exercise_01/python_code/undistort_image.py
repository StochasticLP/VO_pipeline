import math
import numpy as np

from distort_points import distort_points


def undistort_image(img: np.ndarray,
                    K: np.ndarray,
                    D: np.ndarray,
                    bilinear_interpolation: bool = False) -> np.ndarray:
    """
    Corrects an image for lens distortion.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)
        bilinear_interpolation: whether to use bilinear interpolation or not
    """
    H, W, C = np.shape(img)
    undistorted_image = np.zeros((H,W,C))
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = np.vstack((u.flatten(), v.flatten()))
    distorted_x = distort_points(x, D, K)
    
    # Iterate over all pixels
    for i in range(x.shape[1]):

        #bilinear interpolation
        if bilinear_interpolation:
                # Find the four surrounding integer pixel coordinates
                x1, y1 = int(np.floor(distorted_x[0, i])), int(np.floor(distorted_x[1, i]))
                x2, y2 = x1 + 1, y1 + 1

                # Ensure coordinates are within image boundaries
                x1 = np.clip(x1, 0, W - 1)
                x2 = np.clip(x2, 0, W - 1)
                y1 = np.clip(y1, 0, H - 1)
                y2 = np.clip(y2, 0, H - 1)

                # Get the values of the four corner pixels
                q11 = img[y1, x1]
                q12 = img[y2, x1]
                q21 = img[y1, x2]
                q22 = img[y2, x2]

                # Calculate weights based on the fractional part of coordinates
                tx = distorted_x[0, i] - x1
                ty = distorted_x[1, i] - y1
                
                # Perform linear interpolation first in the x-direction
                r1 = q11 * (1 - tx) + q21 * tx
                r2 = q12 * (1 - tx) + q22 * tx

                # Then in the y-direction to get the final interpolated value
                interpolated_value = r1 * (1 - ty) + r2 * ty

                undistorted_image[int(x[1, i]), int(x[0, i])] = interpolated_value
        else:
            u_dist_val = int(round(distorted_x[0, i]))
            v_dist_val = int(round(distorted_x[1, i]))
            undistorted_image[int(x[1, i]), int(x[0, i])] = img[v_dist_val, u_dist_val]

    return undistorted_image / 255

