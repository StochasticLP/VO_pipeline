import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from draw_camera import drawCamera
from reproject_points import reprojectPoints


def side_by_side(fps, filename, translations, quaternions, pts3d, all_pts_2d, K):
    xmin = min(np.min(translations[:,0]), np.min(pts3d[:,0]))
    xmax = max(np.max(translations[:,0]), np.max(pts3d[:,0]))
    ymin = min(np.min(translations[:,1]), np.min(pts3d[:,1]))
    ymax = max(np.max(translations[:,1]), np.max(pts3d[:,1]))
    zmin = min(np.min(translations[:,2]), np.min(pts3d[:,2]))
    zmax = max(np.max(translations[:,2]), np.max(pts3d[:,2]))

    video = None
    
    fig = plt.figure(figsize=(12, 6))

    for i in range(translations.shape[0]):
        fig.clear()
        
        # Left side: 3D trajectory
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_xlim([xmin, xmax])
        ax1.set_ylim([ymin, ymax])
        ax1.set_zlim([zmin, zmax])
        ax1.set_box_aspect((np.ptp(ax1.get_xlim()),
                            np.ptp(ax1.get_ylim()),
                            np.ptp(ax1.get_zlim())))
        ax1.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2])
        rotMat = Rotation.from_quat(quaternions[i, :]).as_matrix()
        drawCamera(ax1, translations[i,:], rotMat, length_scale = 0.1, head_size = 10,
                set_ax_limits = False)
        
        # Right side: image with points
        ax2 = fig.add_subplot(1, 2, 2)
        img_path = "../data/images_undistorted/img_%04d.jpg" % (i + 1)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            R = rotMat.T
            t = -R @ translations[i, :].reshape(3,1)
            M_tilde = np.hstack([R, t])
            p_reproj = reprojectPoints(pts3d, M_tilde, K)
            pts_2d = all_pts_2d[i]
            
            ax2.imshow(img, cmap="gray")
            ax2.scatter(pts_2d[:,0], pts_2d[:,1], marker='o')
            ax2.scatter(p_reproj[:,0], p_reproj[:,1], marker='+')
            
        canvas = FigureCanvas(fig)
        canvas.draw()

        mat = np.array(canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

        if video is None:
            video_size = (mat.shape[1], mat.shape[0])
            video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('X','V','I','D'), 
                fps, video_size)

        mat = cv2.resize(mat, video_size)
        cv2.imshow("test", mat)
        cv2.waitKey(3)
        video.write(mat)

    cv2.destroyAllWindows()
    if video is not None:
        video.release()
