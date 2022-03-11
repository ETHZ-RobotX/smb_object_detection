import yaml
import rospy
import numpy as np

from object_detection.utils import check_validity_lidar2camera_transformation

AXIS_X = 0
AXIS_Y = 1

class PointProjector:
    def __init__(self, config):

        with open(config) as file:
            self.config         = yaml.load(file, Loader=yaml.FullLoader)
            R_camera_lidar      = self.config["R_camera_lidar"]
            R_correction        = self.config["R_correction"]
            t_camera_lidar      = self.config["t_camera_lidar"]
            t_correction        = self.config["t_correction"]
            self.forward_axis   = self.config["forward_axis"]
        
        if not check_validity_lidar2camera_transformation(R_camera_lidar, t_camera_lidar):
            msg = "[pointprojector] Lidar to Camera transformation matrices are not correctly configured. Please check the file %s" %config
            rospy.logerr(msg)
            rospy.signal_shutdown("[pointprojector] Lidar to Camera transformation matrices are not correctly configured!")

        R_camera_lidar = np.float64(R_camera_lidar)
        R_correction = np.float64(R_correction)

        R_camera_lidar = np.matmul(R_camera_lidar,R_correction)       
        t_camera_lidar = np.float64(t_camera_lidar) + np.float64(t_correction)

        self.set_extrinsic_params(R_camera_lidar,t_camera_lidar)
        self.K      = None  
        self.w      = None
        self.h      = None

    def set_intrinsic_params(self, K, size):
        """ Camera Info / Intrinsic parameters setter 
        Args:
            K       : Camera intrinsic matrix 
            size    : Image size
        
        """
        self.K      = K
        self.w      = size[0]
        self.h      = size[1]
        
        self.P = np.zeros((3,4))
        self.P[:,:3] = self.K
       
    def set_extrinsic_params(self, R, t):
        """ Camera extrinsic parameters setter 
        Args:
            R       : Camera extrinsic rotation matrix 
            t       : Camera extrinsic translation matrix 
        """
        self.R = R
        self.t = t

        # T_camera_lidar
        self.T = np.eye(4)
        self.T[:3,:3] = R
        self.T[:3,3] = t
    
    def transform_points(self, points):
        """ Transform points by using the set extrinsic parameters 
        Args:
            points      : Lidar points in source frame

        Returns:
            points      : Lidar points in target frame

        """
        homo_coor = np.ones(points.shape[0])
        XYZ = np.vstack((np.transpose(points),homo_coor))

        XYZ = self.T @ XYZ
        XYZ = XYZ / XYZ[3,:]

        return np.transpose(XYZ[:3,:])

    def project_points(self,points):
        """ Project points on image frame by using the set intrinsic parameters 
        Args:
            points      : Lidar points in camera frame

        Returns:
            points      : Lidar points in pixel coordinates
            indices     : Indices of input lidar points array that are in the front hemisphere

        """
        indices = np.arange(0,len(points))

        # Take only front hemisphere points
        front_hemisphere = points[:, np.abs(self.forward_axis)-1] > 0  if self.forward_axis > 0 else points[:, np.abs(self.forward_axis)-1] < 0
        front_hemisphere_indices = np.nonzero(front_hemisphere)[0]
        indices = indices[front_hemisphere_indices]

        points = points[front_hemisphere_indices, :]
        homo_coor = np.ones(points.shape[0])
        XYZ = np.vstack((np.transpose(points),homo_coor))

        xy = self.P @ XYZ 

        xy = xy / xy[2,None]       

        return np.transpose(xy[:2,:]), indices


    def project_points_on_image(self, points):
        """ Projects 3D points in camera frame onto the image
        Args:
            points: nx3 matrix XYZ in camera frame
            
        Returns:
            points_on_image      : pixel coordinates of projected points
            points_in_FoV_indices: Indices of projected points in camera frame in input points array
        """

        points_on_image, indices = self.project_points(points)
        points_on_image = np.uint32(np.squeeze(points_on_image))
        
        inside_frame_x = np.logical_and((points_on_image[:,AXIS_X] >= 0), (points_on_image[:,AXIS_X] < self.w-1))
        inside_frame_y = np.logical_and((points_on_image[:,AXIS_Y] >= 0), (points_on_image[:,AXIS_Y] < self.h-1))
        inside_frame_indices = np.nonzero(np.logical_and(inside_frame_x,inside_frame_y))[0]

        indices = indices[inside_frame_indices]
        points_on_image = points_on_image[inside_frame_indices,:]    
    
        return points_on_image, indices



