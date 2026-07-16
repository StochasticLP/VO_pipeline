import numpy as np


def pose_vector_to_transformation_matrix(pose_vec: np.ndarray) -> np.ndarray:
    """
    Converts a 6x1 pose vector into a 4x4 transformation matrix.

    Args:
        pose_vec: 6x1 vector representing the pose as [wx, wy, wz, tx, ty, tz]

    Returns:
        T: 4x4 transformation matrix
    """
    w = pose_vec[:3]
    k = w/np.linalg.norm(w)
    k_cross = cross_matrix(k)
    theta = np.linalg.norm(w)

    R = np.identity(3) + np.sin(theta)*k_cross + (1-np.cos(theta))*k_cross@k_cross
    t = np.array(pose_vec[3:]).reshape((3,1))
    Rt = np.hstack((R,t))
    T = np.vstack((Rt,np.array([0,0,0,1]).reshape(1,4)))
    return T

def cross_matrix(v):
    return np.array([[ 0, -v[2], v[1]],
                     [v[2],   0, -v[0]],
                     [-v[1], v[0],  0]])
