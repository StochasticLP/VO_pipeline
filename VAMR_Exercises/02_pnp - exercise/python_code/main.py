import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.transform import Rotation

from estimate_pose_dlt import estimatePoseDLT
from reproject_points import reprojectPoints
from draw_camera import drawCamera
from plot_trajectory_3D import plotTrajectory3D
from side_by_side import side_by_side

def main():
    # Load 
    #    - an undistorted image
    #    - the camera matrix
    #    - detected corners
    image_idx = 1
    undist_img_path = "../data/images_undistorted/img_%04d.jpg" % image_idx
    undist_img = cv2.imread(undist_img_path, cv2.IMREAD_GRAYSCALE)
    if undist_img is None:
        raise ValueError(f"Failed to load image from path: {undist_img_path}")

    K = np.loadtxt("../data/K.txt")
    p_W_corners = 0.01 * np.loadtxt("../data/p_W_corners.txt", delimiter = ",")
    num_corners = p_W_corners.shape[0]

    # Load the 2D projected points that have been detected on the
    # undistorted image into an array
    pts_2d = np.loadtxt("../data/detected_corners.txt").reshape((-1, num_corners, 2))
    # Now that we have the 2D <-> 3D correspondances let's find the camera pose
    # with respect to the world using the DLT algorithm
    M_tilde = estimatePoseDLT(pts_2d[0], p_W_corners, K)

    # Plot the original 2D points and the reprojected points on the image
    p_reproj = reprojectPoints(p_W_corners, M_tilde, K)

    plt.figure()
    plt.imshow(undist_img, cmap = "gray")
    plt.scatter(pts_2d[0,:,0], pts_2d[0,:,1], marker = 'o')
    plt.scatter(p_reproj[:,0], p_reproj[:,1], marker = '+')
    plt.show()

    # Make a 3D plot containing the corner positions and a visualization
    # of the camera axis
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p_W_corners[:,0], p_W_corners[:,1], p_W_corners[:,2]) # type: ignore
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Position of the camera given in the world frame
    R = M_tilde[:, :3]
    pos = -R.T @ M_tilde[:, 3]
    rotMat = R.T

    drawCamera(ax, pos, rotMat, length_scale = 0.1, head_size = 10) # type: ignore
    #drawCamera(ax, np.zeros(3).T, np.eye(3), length_scale = 0.1, head_size = 10)
    plt.show()


def main_video():
    K = np.loadtxt("../data/K.txt")
    p_W_corners = 0.01 * np.loadtxt("../data/p_W_corners.txt", delimiter = ",")
    num_corners = p_W_corners.shape[0]

    all_pts_2d = np.loadtxt("../data/detected_corners.txt").reshape((-1, num_corners, 2))
    num_images = all_pts_2d.shape[0]
    
    M_tildes = np.array([estimatePoseDLT(pts, p_W_corners, K) for pts in all_pts_2d])
    R = M_tildes[:, :, :3]
    rotMats = R.transpose(0, 2, 1)
    translations = -(rotMats @ M_tildes[:, :, 3:4]).squeeze(-1)
    quaternions = Rotation.from_matrix(rotMats).as_quat()

    fps = 30
    filename = "../motion.avi"
    side_by_side(fps, filename, translations, quaternions, p_W_corners, all_pts_2d, K)


if __name__=="__main__":
    # main()
    main_video()
