import os
import yaml
import rospy
import hdbscan
import numpy as np
from scipy.signal import find_peaks

from dataclasses import dataclass

NO_POSE     = -1
MAX_DIST    = 999999

AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

DEFAULT_MAX_OBJECT_DEPTH = 0.25

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

        try:
            self.object_specific_file_dir   = os.path.join(config_dir, config["object_specific_file"])
            with open(self.object_specific_file_dir) as file:
                self.obj_conf               = yaml.load(file, Loader=yaml.FullLoader)
        except:
            rospy.loginfo("[objectlocalizer] Object specific file or path does not exist. It is fine if you are using version 1.")

        self.model_method                   = config["model_method"].lower()
        self.distance_estimator_type        = config["distance_estimator_type"].lower()
        self.distance_estimator_save_data   = config["distance_estimator_save_data"]
        self.bb_contract_percentage         = config["bb_contract_percentage"]
        self.min_cluster_size               = config["min_cluster_size"]
        self.cluster_selection_epsilon      = config["cluster_selection_epsilon"]
        
        self.distance_estimator             = self.estimate_dist_default
        
        if self.distance_estimator_type  != "none":
            
            try:
                with open(os.path.join(config_dir, self.distance_estimator_type + ".yaml" )) as file:
                    self.estimate_dist_cfg  = yaml.load(file, Loader=yaml.FullLoader)
                self.distance_estimator = getattr(self, 'estimate_dist_'+self.distance_estimator_type)
                msg = "[objectlocalizer] Distance estimator " + self.distance_estimator_type + " has been set."
                rospy.loginfo(msg)
            except:
                msg = "[objectlocalizer] Distance estimator " + self.distance_estimator_type + " is not defined. Default one will be used. Check available estimators."
                rospy.logerr(msg)


            if self.distance_estimator_save_data :
                msg1 = "[objectlocalizer] New data will be collected for " + self.distance_estimator_type + " estimator. Please refer to the save data instructions."
                msg2 = "[objectlocalizer] Distance estimator will not be used during the data collection."
                self.data_saver         = getattr(self, 'save_data_'+self.distance_estimator_type)
                self.distance_estimator = self.estimate_dist_default
                rospy.loginfo(msg1)
                rospy.loginfo(msg2)
                
                self.learner_data_dir   = os.path.join(os.path.join(config_dir, "data"), self.distance_estimator_type)
                self.create_save_directory()

        else:
            self.distance_estimator_save_data = False
            rospy.loginfo("[objectlocalizer] Estimator/Learner type is None. No estimator will be used. Data will not be saved.")


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
        
        if self.distance_estimator_type == "bb2dist":
            if not os.path.exists( self.learner_data_dir):
                os.makedirs(self.learner_data_dir)
                 
    def save_data_bb2dist(self, input):
        
        ind, pose = input[0], input[1]
        
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
    
    def estimate_dist_default(self, input):
        """ Default object distance estimator. Returns always 0
        
        Args:
            input          : dummy input

        Returns:
            0
        """
        return 0
    
    def estimate_dist_bb2dist(self, input):
        """ Estimate the object distance with the bb2dist method.
        
        Args:
            input          : list of necessary inputs 
            input[0]       : index of the target object in the input panda data frame 
            input[1]       : class id of the object ex: person, car, bike ...
        
        Returns:
            estimated_dist : Estimated 3D point in camera frame 
        """
        idx, class_id = input[0], input[1]
        p = np.poly1d(self.estimate_dist_cfg[class_id])
        return max(p(self.object_unique_size(idx, self.obj_conf[class_id]['unique'])), 0.5)
            
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

    def method_hdbscan(self,in_BB_3D, obj_class, estimated_dist):
        
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
            indices_ = np.nonzero(cluster.labels_ == -1)[0]
            indices  = np.argmin( np.abs( estimated_dist - in_BB_3D[indices_, AXIS_Z]) )
            avg      = in_BB_3D[indices]
        
        else:  

            distances = np.squeeze(in_BB_3D[indices, AXIS_Z])
            in_range_indices = np.nonzero( ( np.abs(distances - estimated_dist) - min( np.abs(distances - estimated_dist) ) ) < DEFAULT_MAX_OBJECT_DEPTH )[0]

            indices = indices[in_range_indices]   
            avg =  np.mean(in_BB_3D[indices], axis=0)
                             
        return avg, indices

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
        if center_ind == NO_POSE or len(in_BB_indices) < self.min_cluster_size :
            if self.distance_estimator_type == "none":
                new_obj.pt_indices      = np.array([NO_POSE])
                new_obj.pos             = np.array([0,0, NO_POSE])
                new_obj.estimation_type = "none"
            
            else:
                estimated_dist = self.distance_estimator([index, obj_class])

                new_obj.pt_indices      = np.array([NO_POSE])
                new_obj.pos             = self.estimate_pos_with_BB_center(center, estimated_dist)
                new_obj.estimation_type = "estimation"

        else:
            in_BB_3D = self.points3D[in_BB_indices, :]
            # in_BB_2D = self.points2D[in_BB_indices, :]
            
            if self.model_method == "hdbscan":
                try:
                    estimated_dist = self.distance_estimator([index, obj_class])
                except:
                    msg = "[ObjectLocalizer] Estimation failed. There is no data for the object " + obj_class + " !"
                    rospy.logwarn(msg)
                    estimated_dist = 0
                pos, on_object = self.method_hdbscan(in_BB_3D, obj_class, estimated_dist)

            elif self.model_method == "mean":
                on_object = np.arange(0,in_BB_3D.shape[0])
                pos = np.mean(in_BB_3D, axis=0)
            elif self.model_method == "median":
                on_object = np.arange(0,in_BB_3D.shape[0])
                pos = np.median(in_BB_3D, axis=0)
            elif self.model_method == "center":
                on_object = np.arange(0,in_BB_3D.shape[0])
                pos = in_BB_3D[center_ind,:]
            elif self.model_method == "histogram":
                pos, on_object = self.method_histogram(in_BB_3D)
            
            # Allign with the bounding box center
            center_point = in_BB_3D[center_ind]
            center_point[[AXIS_X, AXIS_Y]] = center_point[[AXIS_X, AXIS_Y]] * (pos[AXIS_Z] / center_point[AXIS_Z])
            pos[[AXIS_X, AXIS_Y]] = center_point[[AXIS_X, AXIS_Y]]
            
            new_obj.pt_indices      = np.array([in_BB_indices[on_object]]) if isinstance(on_object, np.int64) else in_BB_indices[on_object]
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
            object_list : list of DetectedObjects correspond to the order of the input objects structure. 

        """

        # Set the data of the scene.
        self.set_scene(objects, points2D, points3D, image)
        self.id_dict = {}

        object_list = []

        # For every object
        for ind in range(len(self.objects)):

            new_obj =  self.get_object_pos(ind)
            object_list.append(new_obj)

            if self.distance_estimator_save_data :
                self.data_saver([ind, new_obj.pos])

        return object_list

