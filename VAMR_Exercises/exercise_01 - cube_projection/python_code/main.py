import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

from pose_vector_to_transformation_matrix import \
    pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized


def main():
    pass

    # load camera poses
    # each row i of matrix 'poses' contains the transformations that transforms
    # points expressed in the world frame to
    # points expressed in the camera frame

    poses = np.loadtxt('data/poses.txt')

    # define 3D corner positions
    # [Nx3] matrix containing the corners of the checkerboard as 3D points
    # (X,Y,Z), expressed in the world coordinate system
    x, y = np.meshgrid(np.arange(0, 9), np.arange(0, 6))
    checkerboard_points = np.vstack((x.flatten(), y.flatten(), np.zeros(54), np.ones(54)))
    checkerboard_points[:2, :] *= 0.04
    # load camera intrinsics
    K = np.loadtxt('data/K.txt')
    D = np.loadtxt('data/D.txt')

    # load one image with a given index
    i = 0
    img = cv2.imread('data/images/img_{:04d}.jpg'.format(i+1))
    #image = cv2.imread('data/images_undistorted/img_0001.jpg')

    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame

    pose_vec = poses[i]
    T = pose_vector_to_transformation_matrix(pose_vec)
        
    # transform 3d points from world to current camera pose
    transformed_points = T @ checkerboard_points
    points_2d = project_points(transformed_points[:3,:], K, D)

    """
    plt.clf()
    plt.close()
    plt.imshow(img.copy(), cmap='gray')
    plt.plot(points_2d[0,:], points_2d[1,:], marker='o', color='r', linestyle='none')
    plt.show()
    """
    
    # undistort image with bilinear interpolation
    start_t = time.time()
    img_undistorted = undistort_image(img, K, D, bilinear_interpolation=True)
    print('Undistortion with bilinear interpolation completed in {}'.format(
       time.time() - start_t))
    
    # vectorized undistortion without bilinear interpolation
    start_t = time.time()
    img_undistorted_vectorized = undistort_image_vectorized(img, K, D)
    print('Vectorized undistortion completed in {}'.format(
        time.time() - start_t))
    
    plt.clf()
    plt.close()
    fig, axs = plt.subplots(2)
    axs[0].imshow(img_undistorted, cmap='gray')
    axs[0].set_axis_off()
    axs[0].set_title('With bilinear interpolation')
    axs[1].imshow(img_undistorted_vectorized, cmap='gray')
    axs[1].set_axis_off()
    axs[1].set_title('Without bilinear interpolation')
    plt.show()
    
    """
    # calculate the cube points to then draw the image
    cube_size = 0.08
    x = 0.08
    y = 0.12
    cube_pts = np.array([
        [x, y, 0],
        [x, y, -cube_size],
        [x+cube_size, y, 0],
        [x+cube_size, y, -cube_size],
        [x, y+cube_size, 0],
        [x, y+cube_size, -cube_size],
        [x+cube_size, y+cube_size, 0],
        [x+cube_size, y+cube_size, -cube_size],
    ])
    # (8 rows 3 columns)
    cube_pts = np.vstack((cube_pts.T, np.ones((1, 8))))
    # (4 rows 8 columns)
    
    cube_pts = T @ cube_pts
    # (8 rows 4 columns)
    cube_pts = project_points(cube_pts[:3,:], K, D).T
    # (8 rows 2 columns)
    
    # Plot the cube
    plt.clf()
    plt.close()
    plt.imshow(img_undistorted, cmap='gray')

    lw = 3

    # base layer of the cube
    plt.plot(cube_pts[[1, 3, 7, 5, 1], 0],
             cube_pts[[1, 3, 7, 5, 1], 1],
             'r-',
             linewidth=lw)

    # top layer of the cube
    plt.plot(cube_pts[[0, 2, 6, 4, 0], 0],
             cube_pts[[0, 2, 6, 4, 0], 1],
             'r-',
             linewidth=lw)

    # vertical lines
    plt.plot(cube_pts[[0, 1], 0], cube_pts[[0, 1], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[2, 3], 0], cube_pts[[2, 3], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[4, 5], 0], cube_pts[[4, 5], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[6, 7], 0], cube_pts[[6, 7], 1], 'r-', linewidth=lw)

    plt.show()
    
    
    # Output video settings
    out_path = 'output_cube_projection.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30

    H, W = img.shape[:2]
    video = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # Process loop
    for i in tqdm(range(len(poses)), desc="Processing images"):
        # Load image
        img_path = f'data/images/img_{i+1:04d}.jpg'
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue

        # Undistort
        img_und = undistort_image_vectorized(img, K, D)

        # Force correct format before writing
        img_und = np.clip(img_und * 255, 0, 255).astype(np.uint8)

        # Get camera pose
        T = pose_vector_to_transformation_matrix(poses[i])

        # Project cube
        cube_cam = T @ cube_pts          # (4,8)
        cube_2d = project_points(cube_cam[:3], K, D).T  # (8,2)

        # Draw cube lines (BGR order)
        color = (0, 0, 255)  # red
        lw = 3

        # bottom face
        pts = cube_2d[[1,3,7,5,1]].astype(int)
        cv2.polylines(img_und, [pts], isClosed=True, color=color, thickness=lw)

        # top face
        pts = cube_2d[[0,2,6,4,0]].astype(int)
        cv2.polylines(img_und, [pts], isClosed=True, color=color, thickness=lw)

        # verticals
        for a, b in [(0,1),(2,3),(4,5),(6,7)]:
            p1 = cube_2d[a].astype(int)
            p2 = cube_2d[b].astype(int)
            cv2.line(img_und, tuple(p1), tuple(p2), color, lw)

        # Write to video
        video.write(img_und)

    video.release()
    print(f"Video saved: {out_path}")
    """
    
if __name__ == "__main__":
    main()
