import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
import cv2

from object_detection.object import Object

class ObjectLocalizer:
    def __init__(self, config):
        self.model_method       = config["model_method"]

    def save_scene(self, objects_BB, points2D, points3D, image=None ):
        """
        Args:
            objects_BB     : 2D object detection results in Panda Dataframe 
            points2D       : 2D Point cloud in camera frame on the image
            points3D       : 3D Point cloud in camera frame 
        """
        self.objects_BB = objects_BB
        self.points3D   = points3D
        self.points2D   = points2D
        self.image      = image
    
    def points_in_BB(self, index, BB_contract_percentage=10 ):
        """
        Args:
            index                   : index of the detected object in Pandas data frame
            BB_contract_percentage  : bounding box contract percentage. E.g. xmin_new = xmin + (xmax-xmin) * BB_contract_percentage / 100 
                    
        Returns:
            inside_BB               : indices of points inside the BB
        """

        x_diff = self.objects_BB['xmax'][index] - self.objects_BB['xmin'][index]
        y_diff = self.objects_BB['ymax'][index] - self.objects_BB['ymin'][index]
        

        inside_BB_x = np.logical_and((self.points2D[:,0] >= self.objects_BB['xmin'][index] + x_diff * BB_contract_percentage / 100 ), \
                                     (self.points2D[:,0] <= self.objects_BB['xmax'][index] - x_diff * BB_contract_percentage / 100))
        inside_BB_y = np.logical_and((self.points2D[:,1] >= self.objects_BB['ymin'][index] + y_diff * BB_contract_percentage / 100), \
                                     (self.points2D[:,1] <= self.objects_BB['ymax'][index] - y_diff * BB_contract_percentage / 100))
        inside_BB = np.argwhere(np.logical_and(inside_BB_x, inside_BB_y)).flatten()

        center = np.array([(self.objects_BB['xmin'][index]+self.objects_BB['xmax'][index])/2.0, \
                  (self.objects_BB['ymin'][index]+self.objects_BB['ymax'][index])/2.0 ])

        center_ind = np.argmin(np.linalg.norm(self.points2D[inside_BB, :] - center, axis=1))

        return inside_BB, center_ind



    def method_hdbscan_color(self, in_BB_2D, in_BB_3D):

        I,J = np.transpose(np.floor(in_BB_2D)).astype(int)
        rgb = self.image[J,I] / 255.0
        features = np.concatenate((in_BB_3D, rgb), axis=1)

        cluster = DBSCAN(eps=0.2, min_samples=1).fit(features)
       
        uniq = np.unique(cluster.labels_)

        min_val = 100000
        indices = None

        for i in uniq:
            indices_ = np.argwhere(cluster.labels_ == i)
            min_val_ = np.mean(np.linalg.norm(in_BB_3D[indices_,:], axis=1))

            if min_val_ < min_val:
                indices = indices_
                min_val = min_val_

        return np.median(in_BB_3D[indices, :], axis=0), indices



    def method_hdbscan(self,in_BB_3D):
       
        cluster = DBSCAN(eps=0.3, min_samples=1).fit(in_BB_3D)
       
        uniq = np.unique(cluster.labels_)

        min_val = 100000
        indices = None

        for i in uniq:
            indices_ = np.argwhere(cluster.labels_ == i)
            min_val_ = np.mean(np.linalg.norm(in_BB_3D[indices_,:], axis=1))

            if min_val_ < min_val:
                indices = indices_
                min_val = min_val_

        return np.mean(in_BB_3D[indices, :],axis=0), indices


    def method_histogram(self,in_BB_3D, method = "distance", bins=100,):

        if method == "distance":
            hist, bin_edges = np.histogram(np.linalg.norm(in_BB_3D, axis=1), bins)
        else:
            hist, bin_edges = np.histogram(in_BB_3D[:,2], bins)

        
        bin_edges = bin_edges[:-1]
        hist = np.insert(hist, 0, 0)
        bin_edges = np.insert(bin_edges, 0, 0)
        peaks, _ = find_peaks(hist)

        """
        plt.plot(bin_edges, hist)
        plt.plot(bin_edges[peaks], hist[peaks], "x")
        plt.savefig('/home/oilter/Downloads/foo.png')
        plt.close()
        """

        inside_peak = np.logical_and((in_BB_3D[:,2] >= bin_edges[peaks[0]-1]), \
                                     (in_BB_3D[:,2] <= bin_edges[peaks[0]+1]))
        
        return np.median(in_BB_3D[inside_peak, :], axis=0), inside_peak
        
        
    def get_object_pos(self, index):
        """        
        Args:
            ind             : index of the detected object in Pandas data frame
            
        Returns:
            pos             : position of the object acc. camera frame
            on_object_ind   : pointcloud-in-frame indices that is on the object 
        """

        in_BB_indices , center_ind= self.points_in_BB(index)
        in_BB_3D = self.points3D[in_BB_indices, :]
        in_BB_2D = self.points2D[in_BB_indices, :]

        on_object = np.arange(0,in_BB_3D.shape[0])

        if self.model_method == "mean":
            pos = np.mean(in_BB_3D, axis=0)
        elif self.model_method == "median":
            pos = np.median(in_BB_3D, axis=0)
        elif self.model_method == "centre":
            pos = in_BB_3D[center_ind,:]
        elif self.model_method == "histogram":
            pos, on_object = self.method_histogram(in_BB_3D)
        elif self.model_method == "hdbscan":
            # pos, on_object = self.method_hdbscan_color(in_BB_2D, in_BB_3D)
            pos, on_object = self. method_hdbscan(in_BB_3D)

        # TODO: Add more method
        # elif method == "median_dist":
        #     distance = np.median(np.linalg.norm(points3D, axis=1))

        return pos, in_BB_indices[on_object]

    def localize(self, objects_BB, points2D, points3D, image=None):
        """
        Args:
            objects_BB     : 2D object detection results in Panda Dataframe 
            points2D       : 2D Point cloud in camera frame on the image
            points3D       : 3D Point cloud in camera frame 
            method         : method to calculate position of object
        
        Returns:
            pos             : nx3 numpy array, position of the objects acc. camera frame
        
        """

        self.save_scene(objects_BB, points2D, points3D, image)
        object_poses = np.empty((0,3))
        on_object_list = []
        for ind in range(len(self.objects_BB)):
            pos, on_object = self.get_object_pos(ind)
            object_poses = np.vstack((object_poses, pos))
            on_object_list.append(np.squeeze(on_object))
        return object_poses, on_object_list

                 
