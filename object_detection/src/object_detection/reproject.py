from PIL.Image import FASTOCTREE
import numpy as np
import cv2

class ImageHandler:
    def __init__(self):
        self.init   = False
        self.K      = None  
        self.w      = None
        self.h      = None

    def set_cameraparams(self, K, shape):
        self.K      = K
        self.w      = shape[0]
        self.h      = shape[1]
        
        self.P = np.zeros((3,4))
        self.P[:,:3] = self.K
       
    def set_transformationparams(self, R, t):
        self.R = R
        self.t = t

        # T_camera_lidar
        self.T = np.eye(4)
        self.T[:3,:3] = R
        self.T[:3,3] = t
    
    def translatePoints(self, points):
        """
        points : nx3 matrix -> X Y Z

        return : nx3 matrix -> X Y Z
        """
        homo_coor = np.ones(points.shape[0])
        XYZ = np.vstack((np.transpose(points),homo_coor))

        XYZ = self.T @ XYZ
        XYZ = XYZ / XYZ[3,:]

        return np.transpose(XYZ[:3,:])

    def projectPoints(self,points):
        """
        points : nx3 matrix -> X Y Z

        return : nx2 matrix -> X Y 
        """
        indices = np.arange(0,len(points))

        # Take only front hemisphere points
        front_hemisphere = points[:, 2] > 0 
        front_hemisphere_indices = np.argwhere(front_hemisphere).flatten()
        indices = indices[front_hemisphere_indices]

        points = points[front_hemisphere_indices, :]
        homo_coor = np.ones(points.shape[0])
        XYZ = np.vstack((np.transpose(points),homo_coor))

        xy = self.P @ XYZ 

        xy = xy / xy[2,None]       

        return np.transpose(xy[:2,:]), indices


    def projectPointsOnImage(self, points):
        """ Projects 3D points onto the image
        Args:
            points: nx3 matrix XYZ
        Returns:
            points_on_image: pixel coordinates of projected points
            points_in_image: 3D coordinates of projected points
        """

        points_on_image, indices = self.projectPoints(points)
        points_on_image = np.uint32(np.squeeze(points_on_image))
        
        inside_frame_x = np.logical_and((points_on_image[:,0] >= 0), (points_on_image[:,0] < self.w))
        inside_frame_y = np.logical_and((points_on_image[:,1] >= 0), (points_on_image[:,1] < self.h))
        inside_frame_indices = np.argwhere(np.logical_and(inside_frame_x,inside_frame_y)).flatten()
        
        indices = indices[inside_frame_indices]
        points_on_image = points_on_image[inside_frame_indices,:]    
    
        return points_on_image, points[indices]



