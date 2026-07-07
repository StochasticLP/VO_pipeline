import numpy as np

from distort_points import distort_points


def undistort_image_vectorized(img: np.ndarray,
                               K: np.ndarray,
                               D: np.ndarray) -> np.ndarray:

    """
    Undistorts an image using the camera matrix and distortion coefficients.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        und_img: undistorted image (HxW)
    """
    H, W = img.shape[:2]
    u, v = np.mgrid[:H, :W][::-1].reshape(2,-1) #2xN

    ud, vd = distort_points(np.vstack((u,v)), D, K)

    ui = np.round(ud).astype(int)
    vi = np.round(vd).astype(int)

    mask = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    und_img = np.zeros_like(img)

    und_img[v[mask], u[mask]] = img[vi[mask], ui[mask]]

    return und_img / 255.0
