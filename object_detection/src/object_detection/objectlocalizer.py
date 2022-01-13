from math import dist
import os
import numpy as np
from dataclasses import dataclass
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN, OPTICS
import yaml
import hdbscan

import rospy

NO_POSE = -1
DATA = "data"

class ObjectLocalizer:
    def __init__(self, config, config_dir):

        self.data_dir                   = os.path.join(config_dir, DATA)
        self.object_specific_file_dir   = os.path.join(config_dir, config["object_specific_file"])
        
        with open(self.object_specific_file_dir) as file:
            self.obj_conf               = yaml.load(file, Loader=yaml.FullLoader)

        self.model_method                   = config["model_method"].lower()
        self.ground_percentage              = config["ground_percentage"]
        self.distance_estimater_type        = config["distance_estimater_type"]
        self.distance_estimater_save_data   = config["distance_estimater_save_data"]

        if self.distance_estimater_type  is not None:
            self.learner_data_dir   = os.path.join(self.data_dir, self.distance_estimater_type)
            self.create_save_directory()
            
            self.estimate_dist_cfg_dir  = os.path.join(config_dir, self.distance_estimater_type + ".yaml" )

            with open(self.estimate_dist_cfg_dir) as file:
                self.estimate_dist_cfg  = yaml.load(file, Loader=yaml.FullLoader)

        else:
            rospy.loginfo("Learner type was not specified. Data will not be saved.")

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
    
    def get_front(self, object0, object1):
        only_in_0 = np.setxor1d(object0.in_BB_indices, object1.in_BB_indices)
        only_in_1 = np.setxor1d(object1.in_BB_indices, object0.in_BB_indices)

        if len(only_in_0) == 0:
            return 0

        if len(only_in_1) == 0:
            return 1 

    def filter_ground(self, point3D, upward):
        """
        Args:
            point3D                   : 3D point translated point cloud 

        Returns:
            non_groun_point3D         : indices of points that are not ground
        """

        if upward < 0:
            indices = np.nonzero(point3D[:,-upward] < max(point3D[:,-upward])*self.ground_percentage)[0]
            
        else:
            indices = np.nonzero(point3D[:,upward] > min(point3D[:,upward])*self.ground_percentage)[0]
        
        return point3D[indices]
               

    def points_in_BB(self, index, contract_percentage_bottom=0, contract_percentage_top=0, contract_percentage_sides=0 ):
        """
        Args:
            index                   : index of the detected object in Pandas data frame
            contract_percentage_*   : bounding box contract percentage of the position *. E.g. xmin_new = xmin + (xmax-xmin) * BB_contract_percentage / 100 
                    
        Returns:
            inside_BB               : indices of points inside the BB
            center_ind              : index of the point that is closest to the center of the object BB
            center                  : pixel coordinates of the center of the object
        """

        x_diff = self.objects_BB['xmax'][index] - self.objects_BB['xmin'][index]
        y_diff = self.objects_BB['ymax'][index] - self.objects_BB['ymin'][index]
        

        inside_BB_x = np.logical_and((self.points2D[:,0] >= self.objects_BB['xmin'][index] + x_diff * contract_percentage_sides / 100 ), \
                                     (self.points2D[:,0] <= self.objects_BB['xmax'][index] - x_diff * contract_percentage_sides / 100))
        inside_BB_y = np.logical_and((self.points2D[:,1] >= self.objects_BB['ymin'][index] + y_diff * contract_percentage_top / 100), \
                                     (self.points2D[:,1] <= self.objects_BB['ymax'][index] - y_diff * contract_percentage_bottom / 100))
        inside_BB = np.argwhere(np.logical_and(inside_BB_x, inside_BB_y)).flatten()

        center = np.array([(self.objects_BB['xmin'][index]+self.objects_BB['xmax'][index])/2.0, \
                  (self.objects_BB['ymin'][index]+self.objects_BB['ymax'][index])/2.0 ])


        if len(inside_BB) == 0:
            inside_BB = np.array([NO_POSE])
            center_ind = NO_POSE
        else:
            center_ind = np.argmin(np.linalg.norm(self.points2D[inside_BB, :] - center, axis=1))

        return inside_BB, center_ind, center

    def method_hdbscan_closeness(self,in_BB_3D, center_id, obj_class, estimated_dist = 0):
        
        cluster = DBSCAN(eps=self.obj_conf[obj_class]["eps"], min_samples=2).fit(in_BB_3D[:,[0,1,2]])
               
        uniq = np.unique(cluster.labels_)

        min_val = 100000
        indices = None
                
        for i in uniq:
            if i == -1:
                continue

            indices_ = np.nonzero(cluster.labels_ == i)[0]
            min_val_ = np.abs( estimated_dist - np.mean(in_BB_3D[indices_,2]) )

            if min_val_ < min_val:
                indices = indices_
                min_val = min_val_

        if indices is None:
            indices = np.array([NO_POSE])
            avg     = np.array([0,0, NO_POSE])
        
        else:          
            
            distances = np.squeeze(in_BB_3D[indices,2])
            in_range_indices = np.nonzero( ( np.abs(distances - estimated_dist) - min( np.abs(distances - estimated_dist) ) ) < self.obj_conf[obj_class]["max_depth"] )[0]

            indices = indices[in_range_indices]

            weights= np.abs(distances[in_range_indices] - max(distances[in_range_indices])) # **2


            if np.sum(weights) == 0:
                weights = weights + 1

            avg =  np.ma.average(in_BB_3D[indices], axis=0, weights=weights)
            
            center_point = in_BB_3D[center_id]
            center_point[:2] = center_point[:2] * (avg[2] / center_point[2])
            avg[:2] = center_point[:2]
                 
        return avg, indices
    
    def method_hdbscan_widht(self,in_BB_3D, center_id, obj_class, estimated_dist = 0):
        
        #cluster = DBSCAN(eps=self.obj_conf[obj_class]["eps"], min_samples=2).fit(in_BB_3D[:,[0,1,2]])
        cluster = hdbscan.HDBSCAN(min_cluster_size=2)
        cluster.fit(in_BB_3D[:,[0,2]])
        uniq = np.unique(cluster.labels_)

        min_val = 100000
        max_width = 0
        indices = None
        
                
        for i in uniq:
            if i == -1:
                continue

            indices_ = np.nonzero(cluster.labels_ == i)[0]
            min_val_ = np.abs( estimated_dist - np.mean(in_BB_3D[indices_,2]) )
            max_width_ = np.abs( max(in_BB_3D[indices_,0]) - min(in_BB_3D[indices_,0]) )

            if min_val_ < 1.5 and max_width_ > max_width :
                max_width = max_width_
                indices = indices_
                #min_val = min_val_

        if indices is None:
            indices = np.array([NO_POSE])
            avg     = np.array([0,0, NO_POSE])
        
        else:          
            
            distances = np.squeeze(in_BB_3D[indices,2])
            in_range_indices = np.nonzero( ( np.abs(distances - estimated_dist) - min( np.abs(distances - estimated_dist) ) ) < self.obj_conf[obj_class]["max_depth"] )[0]

            indices = indices[in_range_indices]

            weights= np.abs(distances[in_range_indices] - max(distances[in_range_indices])) # **2


            if np.sum(weights) == 0:
                weights = weights + 1

            avg =  np.ma.average(in_BB_3D[indices], axis=0, weights=weights)
            
            center_point = in_BB_3D[center_id]
            center_point[:2] = center_point[:2] * (avg[2] / center_point[2])
            avg[:2] = center_point[:2]
                 
        return avg, indices
    
    def method_kMeans(self,in_BB_3D):
       
        cluster = DBSCAN(eps=0.3, min_samples=1).fit(in_BB_3D)
        uniq = np.unique(cluster.labels_)

        min_val = 100000
        indices = None

        for i in uniq:
            indices_ = np.squeeze(np.argwhere(cluster.labels_ == i))
            min_val_ = np.mean(np.linalg.norm(in_BB_3D[indices_], axis=1))

            if min_val_ < min_val:
                indices = indices_
                min_val = min_val_

        if len(indices) > 1:
            return np.mean(in_BB_3D[indices],axis=0), indices
        else:
            return in_BB_3D[indices], indices


    def method_histogram(self,in_BB_3D, method = "distance", bins=100):

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
        
        return np.mean(in_BB_3D[inside_peak, :], axis=0), inside_peak
    
    def get_object_pos(self, index):
        """        
        Args:
            ind             : index of the detected object in Pandas data frame
            
        Returns:
            pos             : position of the object acc. camera frame
            on_object_ind   : pointcloud-in-frame indices that is on the object 
        """
        obj_class = self.objects_BB["name"][index]

        in_BB_indices, center_ind, center = self.points_in_BB(index)

        # If no points falls inside the BB
        if center_ind == NO_POSE :
            on_object_ind = np.array([NO_POSE])
            pos           = np.array([0,0, NO_POSE])

        else:
            in_BB_3D = self.points3D[in_BB_indices, :]
            in_BB_2D = self.points2D[in_BB_indices, :]

            on_object = np.arange(0,in_BB_3D.shape[0])

            estimated_dist = 0
            if self.distance_estimater_type is not None:
                p = np.poly1d(self.estimate_dist_cfg[obj_class])
                estimated_dist = p(self.object_unique_size(index, self.obj_conf[obj_class]['unique']))


            if self.model_method == "hdbscan":
                pos, on_object = self.method_hdbscan_widht(in_BB_3D, center_ind, obj_class, estimated_dist)
            elif self.model_method == "mean":
                pos = np.mean(in_BB_3D, axis=0)
            elif self.model_method == "median":
                pos = np.median(in_BB_3D, axis=0)
            elif self.model_method == "centre":
                pos = in_BB_3D[center_ind,:]
            elif self.model_method == "histogram":
                pos, on_object = self.method_histogram(in_BB_3D)
            
            on_object_ind = in_BB_indices[on_object]

        return pos, on_object_ind

    def localize(self, objects_BB, points2D, points3D, image=None):
        """
        Args:
            objects_BB     : 2D object detection results in Panda Dataframe 
            points2D       : 2D Point cloud in camera frame on the image
            points3D       : 3D Point cloud in camera frame 
            image          : rgb image of detection
        
        Returns:
            object_poses   : n size list of arrays that contains position of the objects acc. camera frame
            on_object_list : n size list of arrays that contains indices of points that fall onto the object for every object

        """

        self.save_scene(objects_BB, points2D, points3D, image)

        object_poses = []
        on_object_list = []

        for ind in range(len(self.objects_BB)):

            pos, on_object = self.get_object_pos(ind)

            if on_object[0] == NO_POSE:
                on_object_list.append(on_object)
            else:
                on_object_list.append(np.array(on_object, dtype=np.int32))

                if self.distance_estimater_save_data :
                    if self.distance_estimater_type== "bb2dist":    
                        self.save_data_bb2dist(ind, pos)
                    else:
                        self.save_data_bb2dist(ind, pos)

            object_poses.append(pos)
            
        return object_poses, on_object_list

    def create_save_directory(self):
        
        if self.distance_estimater_type== "bb2dist":
            if not os.path.exists( self.learner_data_dir):
                os.makedirs(self.learner_data_dir)
                 
    def save_data_bb2dist(self, ind, pose):
        
        for i in range(len(self.objects_BB)): 

            if ind == i :
                continue

            if self.is_overlapping(ind, i):
                return
        
        obj_class = self.objects_BB["name"][ind]
        bb_size = self.object_unique_size(ind, self.obj_conf[obj_class]['unique'])

        txt = obj_class + ".txt"
        input = str(bb_size) + " " + str(pose[2]) + "\n"
        with open( os.path.join( self.learner_data_dir, txt ), 'a' ) as file:
            file.write(input)

    def object_unique_size(self, ind, unique):
        if unique == 'x':
            size = self.objects_BB['xmax'][ind] - self.objects_BB['xmin'][ind]
        elif unique == 'y':
            size = self.objects_BB['ymax'][ind] - self.objects_BB['ymin'][ind]
        else :
            min_p = np.array( [self.objects_BB['xmin'][ind], self.objects_BB['ymin'][ind]] )
            max_p = np.array( [self.objects_BB['xmax'][ind], self.objects_BB['ymax'][ind]] )
            size = np.linalg.norm(max_p - min_p)

        return size

    def is_overlapping(self, ind1, ind2):
        return (self.objects_BB['xmax'][ind1] >= self.objects_BB['xmin'][ind2]) and \
               (self.objects_BB['xmax'][ind2] >= self.objects_BB['xmin'][ind1]) and \
               (self.objects_BB['ymax'][ind1] >= self.objects_BB['ymin'][ind2]) and \
               (self.objects_BB['ymax'][ind2] >= self.objects_BB['ymin'][ind1]) 