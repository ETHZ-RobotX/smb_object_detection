import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import cv2

from object_detection.object import Object

class ObjectLocalizer:
    def __init__(self, config):
        self.model_method       = config["model_method"]

    def save_scene(self, objects_BB, points2D, points3D ):
        """
        Args:
            objects_BB     : 2D object detection results in Panda Dataframe 
            points2D       : 2D Point cloud in camera frame on the image
            points3D       : 3D Point cloud in camera frame 
        """
        self.objects_BB = objects_BB
        self.points3D   = points3D
        self.points2D   = points2D
    
    def points_in_BB(self,index):
        """
        Args:
            ind             : index of the detected object in Pandas data frame
                    
        Returns:
            inside_BB       : indices of points inside the BB
        """
        
        inside_BB_x = np.logical_and((self.points2D[:,0] >= self.objects_BB['xmin'][index]), \
                                     (self.points2D[:,0] <= self.objects_BB['xmax'][index]))
        inside_BB_y = np.logical_and((self.points2D[:,1] >= self.objects_BB['ymin'][index]), \
                                     (self.points2D[:,1] <= self.objects_BB['ymax'][index]))
        inside_BB = np.argwhere(np.logical_and(inside_BB_x, inside_BB_y)).flatten()

        center = np.array([(self.objects_BB['xmin'][index]+self.objects_BB['xmax'][index])/2.0, \
                  (self.objects_BB['ymin'][index]+self.objects_BB['ymax'][index])/2.0 ])

        center_ind = (np.abs(self.points2D - center)).argmin()

        return inside_BB, center_ind
    
    def method_histogram(self,in_BB_3D):
        hist, bin_edges = np.histogram(in_BB_3D[:,2], 100)
        bin_edges = bin_edges[:-1]
        hist = np.insert(hist, 0, 0)
        bin_edges = np.insert(bin_edges, 0, 0)
        peaks, _ = find_peaks(hist)
        
        plt.plot(bin_edges, hist)
        plt.plot(bin_edges[peaks], hist[peaks], "x")
        plt.savefig('/home/oilter/Downloads/foo.png')
        plt.close()

        inside_peak = np.logical_and((in_BB_3D[:,2] >= bin_edges[peaks[0]]), \
                                     (in_BB_3D[:,2] <= bin_edges[peaks[0]+1]))
        
        return np.median(in_BB_3D[inside_peak, :], axis=0)
        
        

    def get_object_pos(self, index):
        """        
        Args:
            ind             : index of the detected object in Pandas data frame
            
        Returns:
            pos             : position of the object acc. camera frame
        """

        indices , center_ind= self.points_in_BB(index)
        in_BB_3D = self.points3D[indices, :]

        if self.model_method == "mean":
            pos = np.mean(in_BB_3D, axis=0)
        elif self.model_method == "median":
            pos = np.median(in_BB_3D, axis=0)
        elif self.model_method == "centre":
            pos = self.points3D[center_ind]
        elif self.model_method == "histogram":
            pos = self.method_histogram(in_BB_3D)
        
        # TODO: Add more method
        # elif method == "median_dist":
        #     distance = np.median(np.linalg.norm(points3D, axis=1))

        return pos, indices

    def localize(self, objects_BB, points2D, points3D, method=None):
        """
        Args:
            objects_BB     : 2D object detection results in Panda Dataframe 
            points2D       : 2D Point cloud in camera frame on the image
            points3D       : 3D Point cloud in camera frame 
            method         : method to calculate position of object
        
        Returns:
            pos             : nx3 numpy array, position of the objects acc. camera frame
        
        """
        if method is not None:
            self.model_method = method

        self.save_scene(objects_BB, points2D, points3D)
        object_poses = np.empty((0,3))
        indices_list = []
        for ind in range(len(self.objects_BB)):
            pos, indices = self.get_object_pos(ind)
            object_poses = np.vstack((object_poses, pos))
            indices_list.append(indices)
        return object_poses, indices_list

                 
