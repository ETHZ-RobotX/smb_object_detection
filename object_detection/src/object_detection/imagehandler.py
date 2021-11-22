from PIL.Image import FASTOCTREE
import numpy as np
import cv2

class ImageHandler:
    def __init__(self):
        self.init   = False
        self.K      = None
        self.dist   = None
        self.w      = None
        self.h      = None
        self.new_K  = None
        self.roi    = None

    def set_cameraparams(self, K, dist, shape):
        self.K      = K
        self.dist   = dist
        self.w      = shape[0]
        self.h      = shape[1]
        
        self.new_K, self.roi = cv2.getOptimalNewCameraMatrix( self.K,
                                                              self.dist,
                                                              (self.w,self.h),
                                                              1,
                                                              (self.w,self.h))

        self.P = np.zeros((3,4))
        self.P[:,:3] = self.new_K

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.dist, np.eye(3), self.new_K, (self.w,self.h),cv2.CV_16SC2)
        


    def set_transformationparams(self, R, t):
        self.R = R
        self.t = t

        # T_camera_lidar
        self.T = np.eye(4)
        self.T[:3,:3] = R
        self.T[:3,3] = t

    def undistort(self, img):
        
        # dst = cv2.undistort(img, self.K, self.dist, self.new_K)
        
        #crop the image
        # x,y,w,h = self.roi
        # dst = dst[y:y+h, x:x+w]
        # 

        # dst = cv2.fisheye.undistortImage(img, self.K, self.dist)
        dst = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return dst
    
    def __undistortPoints(self, points):
        dst = cv2.undistortPoints(np.float64(points), self.K, self.dist, self.new_K )
        return dst
    
    def __translatePoints(self, points):
        """
        points : nx3 matrix -> X Y Z

        return : nx3 matrix -> X Y Z
        """
        homo_coor = np.ones(points.shape[0])
        XYZ = np.vstack((np.transpose(points),homo_coor))

        XYZ = self.T @ XYZ
        XYZ = XYZ / XYZ[3,:]

        return np.transpose(XYZ[:3,:])

    def __projectPoints(self,points):
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


    def projectPoints(self, points, use_cv2=False):
        
        if use_cv2:
            indices = np.arange(0,len(points))
            front_hemisphere = points[:, 1] < 0 
            front_hemisphere_indices = np.argwhere(front_hemisphere).flatten()
            indices = indices[front_hemisphere_indices]
            points = points[front_hemisphere_indices,:]
            points_on_image = cv2.projectPoints(points, self.R, self.t, self.K, self.dist)[0]
            
        else:
            translated_points = self.__translatePoints(points)
            points_on_image, indices = self.__projectPoints(translated_points)
          
        points_on_image = np.uint32(np.squeeze(points_on_image))
        
        inside_frame_x = np.logical_and((points_on_image[:,0] >= 0), (points_on_image[:,0] < self.w))
        inside_frame_y = np.logical_and((points_on_image[:,1] >= 0), (points_on_image[:,1] < self.h))
        inside_frame_indices = np.argwhere(np.logical_and(inside_frame_x,inside_frame_y)).flatten()
        
        indices = indices[inside_frame_indices]
        points_on_image = points_on_image[inside_frame_indices,:]    

        return points_on_image, indices

    def projectPoints2(self, points):
        
        front_hemisphere = points[:, 1] < 0 
        front_hemisphere_indices = np.argwhere(front_hemisphere).flatten()
        points_front = points[front_hemisphere_indices,:]
        points_on_image_cv2 = cv2.projectPoints(points_front, self.R, self.t, self.K, self.dist)[0]
            
        
        translated_points = self.__translatePoints(points)
        points_on_image_classic = self.__projectPoints(translated_points)
          
        
        points_on_image_classic = np.uint32(np.squeeze(points_on_image_classic))
        inside_frame_x = np.logical_and((points_on_image_classic[:,0] >= 0), (points_on_image_classic[:,0] < self.w))
        inside_frame_y = np.logical_and((points_on_image_classic[:,1] >= 0), (points_on_image_classic[:,1] < self.h))
        inside_frame_indices = np.argwhere(np.logical_and(inside_frame_x,inside_frame_y)).flatten()
        points_on_image_classic = points_on_image_classic[inside_frame_indices,:]       


        points_on_image_cv2 = np.uint32(np.squeeze(points_on_image_cv2))
        inside_frame_x = np.logical_and((points_on_image_cv2[:,0] >= 0), (points_on_image_cv2[:,0] < self.w))
        inside_frame_y = np.logical_and((points_on_image_cv2[:,1] >= 0), (points_on_image_cv2[:,1] < self.h))
        inside_frame_indices = np.argwhere(np.logical_and(inside_frame_x,inside_frame_y)).flatten()
        points_on_image_cv2 = points_on_image_cv2[inside_frame_indices,:]    

        return points_on_image_cv2, points_on_image_classic


