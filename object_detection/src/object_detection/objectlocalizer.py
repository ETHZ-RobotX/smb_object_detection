import os
import yaml
import rospy
import hdbscan
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks

from dataclasses import dataclass

NO_POSE     = -1
MAX_DIST    = 999999

AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

@dataclass
class DetectedObject:
    """Class for keeping the info of objects"""
    id              : int
    idx             : int
    pos             : np.array
    pt_indices      : np.array
    estimation_type : str

class ObjectLocalizer:
    def __init__(self, config, config_dir):

        self.object_specific_file_dir   = os.path.join(config_dir, config["object_specific_file"])
        
        with open(self.object_specific_file_dir) as file:
            self.obj_conf               = yaml.load(file, Loader=yaml.FullLoader)

        self.model_method                   = config["model_method"].lower()
        self.distance_estimater_type        = config["distance_estimater_type"].lower()
        self.distance_estimater_save_data   = config["distance_estimater_save_data"]
        self.bb_contract_percentage         = config["bb_contract_percentage"]
        self.min_cluster_size               = config["min_cluster_size"]
        self.cluster_selection_epsilon      = config["cluster_selection_epsilon"]
        

        if self.distance_estimater_type  != "none":
            self.data_dir           = os.path.join(config_dir, "data")
            self.learner_data_dir   = os.path.join(self.data_dir, self.distance_estimater_type)
            self.create_save_directory()
            
            self.estimate_dist_cfg_dir  = os.path.join(config_dir, self.distance_estimater_type + ".yaml" )

            with open(self.estimate_dist_cfg_dir) as file:
                self.estimate_dist_cfg  = yaml.load(file, Loader=yaml.FullLoader)

        else:
            rospy.loginfo("Learner type was not specified. Data will not be saved.")

    def set_scene(self, objects, points2D, points3D, image=None ):
        """Set the scene info such as objects, points2D, points3D, image.
        
        Args:
            objects     : 2D object detection results in Panda Dataframe 
            points2D    : 2D Point cloud in camera frame on the image
            points3D    : 3D Point cloud in camera frame 
        """
        self.objects    = objects
        self.points3D   = points3D
        self.points2D   = points2D
        self.image      = image

    def set_intrinsic_camera_param(self, K):
        """Set intrinsic camera parameters.

        Args:
            K     : intrinsic camera parameters
        """
        self.K = K
    
    def create_save_directory(self):
        
        if self.distance_estimater_type == "bb2dist":
            if not os.path.exists( self.learner_data_dir):
                os.makedirs(self.learner_data_dir)
                 
    def save_data_bb2dist(self, ind, pose):
        
        for i in range(len(self.objects)): 

            if ind == i :
                continue

            if self.is_overlapping(ind, i):
                return
        
        obj_class = self.objects["name"][ind]
        bb_size = self.object_unique_size(ind, self.obj_conf[obj_class]['unique'])

        txt = obj_class + ".txt"
        input = str(bb_size) + " " + str(pose[2]) + "\n"
        with open( os.path.join( self.learner_data_dir, txt ), 'a' ) as file:
            file.write(input)

    def object_unique_size(self, ind, unique):
        if unique == 'x':
            size = self.objects['xmax'][ind] - self.objects['xmin'][ind]
        elif unique == 'y':
            size = self.objects['ymax'][ind] - self.objects['ymin'][ind]
        else :
            min_p = np.array( [self.objects['xmin'][ind], self.objects['ymin'][ind]] )
            max_p = np.array( [self.objects['xmax'][ind], self.objects['ymax'][ind]] )
            size = np.linalg.norm(max_p - min_p)

        return size

    def is_overlapping(self, ind1, ind2):
        return (self.objects['xmax'][ind1] >= self.objects['xmin'][ind2]) and \
               (self.objects['xmax'][ind2] >= self.objects['xmin'][ind1]) and \
               (self.objects['ymax'][ind1] >= self.objects['ymin'][ind2]) and \
               (self.objects['ymax'][ind2] >= self.objects['ymin'][ind1]) 

    def object_id(self, class_id):
        """Returns the object unique id in the scene.

        Args:
            class_id    : class_id (string) 
        
        Returns:
            object_id   : id according to previous occurrences. 
        """

        if class_id in self.id_dict:
            object_id = self.id_dict[class_id] + 1
            self.id_dict[class_id] += 1
        else:
            object_id = 0
            self.id_dict[class_id] = 0
        return object_id 

    def estimate_pos_with_BB_center(self, center, est_dist):
        """ Estimate the object position with the center of the BB and an estimated distance.
        
        Args:
            center          : Center pixel coordinates of the BB
            est_dist        : Estimated distance of the BB
        
        Returns:
            estimated_pos   : Estimated 3D point in camera frame 
        """

        X = ( center[0] - self.K[0,2] ) * est_dist / self.K[0,0]
        Y = ( center[1] - self.K[1,2] ) * est_dist / self.K[1,1]

        return [X, Y, est_dist]
    
    def estimate_dist_bb2dist(self, idx, class_id):
        """ Estimate the object distance with the bb2dist method.
        
        Args:
            idx            : index of the target object in the input panda data frame 
            class_id       : class id of the object ex: person, car, bike ...
        
        Returns:
            estimated_dist : Estimated 3D point in camera frame 
        """
        p = np.poly1d(self.estimate_dist_cfg[class_id])
        return p(self.object_unique_size(idx, self.obj_conf[class_id]['unique']))
            
    def points_in_BB(self, index):
        """ Finds the 3D/2D point indices that fall into BB of given object by its index, along with the center point index and pos. 

        Args:
            index                   : index of the detected object in Pandas data frame
                    
        Returns:
            inside_BB               : indices of points inside the BB
            center_ind              : index of the point that is closest to the center of the object BB
            center                  : pixel coordinates of the center of the object
        """

        x_diff = self.objects['xmax'][index] - self.objects['xmin'][index]
        y_diff = self.objects['ymax'][index] - self.objects['ymin'][index]
        
        inside_BB_x = np.logical_and((self.points2D[:,0] >= self.objects['xmin'][index] + x_diff * self.bb_contract_percentage / 100 ), \
                                     (self.points2D[:,0] <= self.objects['xmax'][index] - x_diff * self.bb_contract_percentage / 100))
        inside_BB_y = np.logical_and((self.points2D[:,1] >= self.objects['ymin'][index] + y_diff * self.bb_contract_percentage / 100), \
                                     (self.points2D[:,1] <= self.objects['ymax'][index] - y_diff * self.bb_contract_percentage / 100))
        inside_BB = np.nonzero(np.logical_and(inside_BB_x, inside_BB_y))[0]

        center = np.array([(self.objects['xmin'][index]+self.objects['xmax'][index])/2.0, \
                  (self.objects['ymin'][index]+self.objects['ymax'][index])/2.0 ])

        if len(inside_BB) == 0:
            inside_BB = np.array([NO_POSE])
            center_ind = NO_POSE
        else:
            center_ind = np.argmin(np.linalg.norm(self.points2D[inside_BB, :] - center, axis=1))

        return inside_BB, center_ind, center

    def method_hdbscan_closeness(self,in_BB_3D, center_id, obj_class, estimated_dist):
        
        cluster = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size , \
                                  cluster_selection_epsilon=self.cluster_selection_epsilon).fit(in_BB_3D[:,[AXIS_X, AXIS_Z]])
        unique = np.unique(cluster.labels_)

        min_val = MAX_DIST
        indices = None
                
        for i in unique:
            if i == -1:
                continue

            indices_ = np.nonzero(cluster.labels_ == i)[0]
            min_val_ = np.abs( estimated_dist - min(in_BB_3D[indices_,AXIS_Z]) )

            if min_val_ < min_val:
                indices = indices_
                min_val = min_val_

        if indices is None:
            print("in None")
            indices_ = np.nonzero(cluster.labels_ == -1)[0]
            indices  = np.argmin( np.abs( estimated_dist - in_BB_3D[indices_, AXIS_Z]) )
            avg      = in_BB_3D[indices]
        
        else:          
            
            distances = np.squeeze(in_BB_3D[indices, AXIS_Z])
            # in_range_indices = np.nonzero( ( np.abs(distances - estimated_dist) - min( np.abs(distances - estimated_dist) ) ) < self.obj_conf[obj_class]["max_depth"] )
            in_range_indices = np.nonzero( np.abs( min(distances) - distances ) < self.obj_conf[obj_class]["max_depth"] )[0]

            indices = indices[in_range_indices]   
            avg =  np.mean(in_BB_3D[indices], axis=0)
            
        center_point = in_BB_3D[center_id]
        center_point[[AXIS_X, AXIS_Y]] = center_point[[AXIS_X, AXIS_Y]] * (avg[AXIS_Z] / center_point[AXIS_Z])
        avg[[AXIS_X, AXIS_Y]] = center_point[[AXIS_X, AXIS_Y]]
                 
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

        obj_class = self.objects["name"][index]
        
        # Id of the object 
        new_obj_id = self.object_id(obj_class)

        # New Object data object
        new_obj = DetectedObject( id=new_obj_id, idx=index, pos = None, pt_indices=None, estimation_type=None )

        in_BB_indices, center_ind, center = self.points_in_BB(index)

        # If no points falls inside the BB
        if center_ind == NO_POSE :

            if self.distance_estimater_type == "none":
                new_obj.pt_indices      = np.array([NO_POSE])
                new_obj.pos             = np.array([0,0, NO_POSE])
                new_obj.estimation_type = "none"
            
            else:
                estimated_dist = self.estimate_dist_bb2dist(index, obj_class)

                new_obj.pt_indices      = np.array([center_ind])
                new_obj.pos             = self.estimate_pos_with_BB_center(center, estimated_dist)
                new_obj.estimation_type = "estimation"

        else:
            in_BB_3D = self.points3D[in_BB_indices, :]
            # in_BB_2D = self.points2D[in_BB_indices, :]
            
            if self.model_method == "hdbscan":

                estimated_dist = 0
                if self.distance_estimater_type != "bb2dist":
                    estimated_dist = self.estimate_dist_bb2dist(index, obj_class)

                pos, on_object = self.method_hdbscan_closeness(in_BB_3D, center_ind, obj_class, estimated_dist)

            elif self.model_method == "mean":
                on_object = np.arange(0,in_BB_3D.shape[0])
                pos = np.mean(in_BB_3D, axis=0)
            elif self.model_method == "median":
                on_object = np.arange(0,in_BB_3D.shape[0])
                pos = np.median(in_BB_3D, axis=0)
            elif self.model_method == "centre":
                on_object = np.arange(0,in_BB_3D.shape[0])
                pos = in_BB_3D[center_ind,:]
            elif self.model_method == "histogram":
                pos, on_object = self.method_histogram(in_BB_3D)
            
            new_obj.pt_indices      = in_BB_indices[on_object]
            new_obj.pos             = pos
            new_obj.estimation_type = "measurement"

        return new_obj

    def localize(self, objects, points2D, points3D, image=None):
        """
        Args:
            objects     : 2D object detection results in Panda Dataframe 
            points2D    : 2D Point cloud in camera frame on the image
            points3D    : 3D Point cloud in camera frame 
            image       : RGB image of detection
        
        Returns:
            object_poses   : n size list of arrays that contains position of the objects acc. camera frame
            on_object_list : n size list of arrays that contains indices of points that fall onto the object for every object

        """

        # Set the data of the scene.
        self.set_scene(objects, points2D, points3D, image)
        self.id_dict = {}

        object_list = []

        # For every object
        for ind in range(len(self.objects)):

            new_obj =  self.get_object_pos(ind)
            object_list.append(new_obj)

            if self.distance_estimater_save_data :
                if self.distance_estimater_type == "bb2dist":    
                    self.save_data_bb2dist(ind, new_obj.pos)
                else:
                    msg = "Estimater type " + self.distance_estimater_type + " is not implemented."
                    rospy.logerr(msg)

        return object_list

