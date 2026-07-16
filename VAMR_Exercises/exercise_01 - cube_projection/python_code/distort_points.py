import numpy as np


def distort_points(x: np.ndarray,
                   D: np.ndarray,
                   K: np.ndarray) -> np.ndarray:
    """
    Applies lens distortion to 2D points xon the image plane.

    Args:
        x: 2d points (Nx2)
        D: distortion coefficients (4x1)
        K: camera matrix (3x3)
    """
    k1, k2 = D[:2]
    c = K[:2, 2].reshape(2, 1)
    xn = x - c
    r2 = np.sum(xn**2, axis=0)
    return xn * (1 + k1*r2 + k2*r2**2) + c
