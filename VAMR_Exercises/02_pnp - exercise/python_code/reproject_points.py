import numpy as np

def reprojectPoints(P, M_tilde, K):
    # Reproject 3D points given a projection matrix
    #
    # P         [n x 3] coordinates of the 3d points in the world frame
    # M_tilde   [3 x 4] projection matrix
    # K         [3 x 3] camera matrix
    #
    # Returns [n x 2] coordinates of the reprojected 2d points

    P = np.hstack([P, np.ones((P.shape[0],1))])
    p_reproj = K @ M_tilde @ P.T
    p_reproj = p_reproj / p_reproj[2, :]
    return p_reproj.T[:, :2]
