import numpy as np
import cv2


def points_in_BB(object, points2D, ind):
    """
           
    Args:
        object          : object infos in Pandas data frame
        points2D        : lidar points inside bounding box on image
        points3D        : lidar points inside bounding box in 3D
    
    Returns:
        on_object       : indices of points inside the BB

    """

    inside_BB_x = np.logical_and((points2D[:,0] >= object['xmin'][ind]), (points2D[:,0] <= object['xmax'][ind]))
    inside_BB_y = np.logical_and((points2D[:,1] >= object['ymin'][ind]), (points2D[:,1] <= object['ymax'][ind]))
    inside_BB = np.argwhere(np.logical_and(inside_BB_x, inside_BB_y)).flatten()

    return inside_BB

def points_on_object(object, points2D, points3D):
    """
           
    Args:
        object          : object infos in Pandas data frame
        points2D        : lidar points inside bounding box on image
        points3D        : lidar points inside bounding box in 3D
    
    Returns:
        object_points   : lidar points on the object in 3D  

    """

    inside_BB = points_in_BB(object, points2D)
    inside_BB_3D = points3D[inside_BB]