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

    def set_transformationparams(self, R, t):
        self.R = R
        self.t = t

    def undistort(self, img):
        dst = cv2.undistort(img, self.K, self.dist, None, self.new_K)
        
        #crop the image
        # x,y,w,h = self.roi
        # dst = dst[y:y+h, x:x+w]

        return dst

    def projectPoints(self, points):
        
        front_hemisphere = points[:, 1] < 0 
        front_hemisphere_indices = np.argwhere(front_hemisphere).flatten()      
        
        points = points[front_hemisphere_indices,:]
            
        imgPts = cv2.projectPoints(points, self.R, self.t, self.K, self.dist)[0]
        imgPts = np.uint32(np.squeeze(imgPts))

        inside_frame_x = np.logical_and((imgPts[:,0] >= 0), (imgPts[:,0] < self.w))
        inside_frame_y = np.logical_and((imgPts[:,1] >= 0), (imgPts[:,1] < self.h))
        inside_frame_indices = np.argwhere(np.logical_and(inside_frame_x,inside_frame_y)).flatten()

        imgPts = imgPts[inside_frame_indices,:]  


        return imgPts
