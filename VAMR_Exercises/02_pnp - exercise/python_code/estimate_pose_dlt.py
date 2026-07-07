import numpy as np

def estimatePoseDLT(p, P, K):
    # Estimates the pose of a camera using a set of 2D-3D correspondences
    # and a given camera matrix.
    # 
    # p  [n x 2] array containing the undistorted coordinates of the 2D points
    # P  [n x 3] array containing the 3D point positions
    # K  [3 x 3] camera matrix
    #
    # Returns a [3 x 4] projection matrix of the form 
    #           M_tilde = [R_tilde | alpha * t] 
    # where R is a rotation matrix. M_tilde encodes the transformation 
    # that maps points from the world frame to the camera frame

    # Convert 2D to normalized coordinates
    K_inv = np.linalg.inv(K)
    p_bar = K_inv @ np.vstack([p.T, np.ones(p.shape[0])])

    # Build measurement matrix Q

    # Add ones to P for homogenous coordinates
    P = np.hstack([P, np.ones((P.shape[0],1))])

    u = p_bar[0, :]
    v = p_bar[1, :]
    P_u = P*-u[:,None]
    P_v = P*-v[:,None]
    zeros = np.zeros((len(p),4))

    A1 = np.hstack([P, zeros, P_u])
    A2 = np.hstack([zeros, P, P_v])

    #Concatenate matrices
    Q = np.vstack([A1, A2])

    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    U, s, Vh = np.linalg.svd(Q)
    M = Vh[-1, :]
    if M[-1] < 0:
        M *= -1
    M = np.reshape(M, (3,4))
    
    # Extract [R | t] with the correct scale

    R = M[:, :3]

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    U, s, Vh = np.linalg.svd(R)
    R_tilde = U @ Vh

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix
    alpha = np.linalg.norm(R_tilde, 'fro') / np.linalg.norm(R)

    # Build M_tilde with the corrected rotation and scale
    M_tilde = np.hstack([R_tilde, alpha*M[:, 3:]])

    return M_tilde
