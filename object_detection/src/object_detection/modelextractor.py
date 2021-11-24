import numpy as np
import cv2


def points_in_BB(object, ind, points2D):
    """
           
    Args:
        object          : object infos in Pandas data frame
        ind             : index of the detected object in Pandas data frame
        points2D        : lidar points inside bounding box on image
        
    
    Returns:
        on_object       : indices of points inside the BB

    """

    inside_BB_x = np.logical_and((points2D[:,0] >= object['xmin'][ind]), (points2D[:,0] <= object['xmax'][ind]))
    inside_BB_y = np.logical_and((points2D[:,1] >= object['ymin'][ind]), (points2D[:,1] <= object['ymax'][ind]))
    inside_BB = np.argwhere(np.logical_and(inside_BB_x, inside_BB_y)).flatten()

    return inside_BB

def distance2object(points3D, method="mean_dist"):
    """
    Args:
        points3D        : lidar points inside bounding box in 3D
        method          : the way to calculate points on the 
    
    Returns:
        distance        : distance to object 
    """

    if method == "mean_dist":
        distance = np.mean(np.linalg.norm(points3D, axis=1))
    elif method == "mean_z":
        distance = np.mean(points3D[:,2])
    elif method == "median_dist":
        distance = np.median(np.linalg.norm(points3D, axis=1))
    elif method == "median_z":
        distance = np.median(points3D[:,2])

    return distance


