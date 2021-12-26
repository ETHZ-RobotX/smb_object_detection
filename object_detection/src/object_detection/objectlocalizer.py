import os
import numpy as np
from dataclasses import dataclass
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
import yaml

NO_POSE = -1

@dataclass
class BoundingBox:
    min_x : int
    min_y : int
    max_x : int
    max_y : int

@dataclass
class Object:
    object_class  : str
    in_BB_indices : np.array
    in_BB_center_index : int
    BB : BoundingBox


class ObjectLocalizer:
    def __init__(self, config, data_dir, data_save):

        self.data_save = data_save
        self.data_dir = data_dir

        with open(config) as file:
            self.config             = yaml.load(file, Loader=yaml.FullLoader)
            self.obj_conf           = self.config['objects']
            self.model_method       = self.config["model_method"].lower()
            self.ground_percentage  = self.config["ground_percentage"]
            self.use_estimated_dist = self.config["self.use_estimated_dist"]

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
               

    def points_in_BB(self, index, contract_percentage_bottom=10, contract_percentage_top=10, contract_percentage_sides=10 ):
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
        
        cluster = DBSCAN(eps=self.obj_conf[obj_class]["eps"], min_samples=2).fit(in_BB_3D[:,[0,2]])
        uniq = np.unique(cluster.labels_)

        min_val = 100000
        indices = None

        # indices = np.nonzero(in_BB_3D[:,1] > 0)[0]
        # avg = [0,0,0]
                
        for i in uniq:
            if i == -1:
                continue

            indices_ = np.nonzero(cluster.labels_ == i)[0]
            min_val_ = np.abs( estimated_dist - np.mean(in_BB_3D[indices_,2]) )

            if min_val_ < min_val:
                indices = indices_
                min_val = min_val_

        
        distances = np.squeeze(in_BB_3D[indices,2])
        in_range_indices = np.nonzero( ( distances - min(distances) ) < self.obj_conf[obj_class]["max_depth"] )[0]

        indices = indices[in_range_indices]
        weights = np.abs(distances[in_range_indices] - max(distances[in_range_indices])) # **2

        if np.sum(weights) == 0:
            weights = weights + 1

        avg =  np.ma.average(in_BB_3D[indices], axis=0, weights=weights)
        
        center_point = in_BB_3D[center_id]
        center_point[:2] = center_point[:2] * (avg[2] / center_point[2])
        avg[:2] = center_point[:2]
         
        print("Estimated dist: ", estimated_dist, "Pose: ", avg[2])
        
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

    def method_hdbscan(self,in_BB_3D):
       
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

        in_BB_indices, center_ind, center = self.points_in_BB(index, \
                                            contract_percentage_bottom = self.obj_conf[obj_class]['contract_percentage_bottom'], \
                                            contract_percentage_top    = self.obj_conf[obj_class]['contract_percentage_top'], \
                                            contract_percentage_sides  = self.obj_conf[obj_class]['contract_percentage_sides'])

        # If no points falls inside the BB
        if center_ind == NO_POSE :
            on_object_ind = np.array([NO_POSE])
            pos           = np.array([0,0, NO_POSE])
            center_ind = NO_POSE

        else:
            in_BB_3D = self.points3D[in_BB_indices, :]
            in_BB_2D = self.points2D[in_BB_indices, :]

            on_object = np.arange(0,in_BB_3D.shape[0])

            # estimated_dist = 5.82383393 - 0.00956915 * self.object_unique_size(index, self.obj_conf[obj_class]['unique'])
            estimated_dist = 0
            if self.use_estimated_dist:
                p = np.poly1d([-1.33652937e-13,  4.55267531e-10, -5.76408389e-07,  3.45532171e-04, -1.03922962e-01,  1.50021037e+01])
                estimated_dist = p(self.object_unique_size(index, self.obj_conf[obj_class]['unique']))


            if self.model_method == "hdbscan":
                pos, on_object = self.method_hdbscan_closeness(in_BB_3D, center_ind, obj_class, estimated_dist)
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
                on_object_list.append(np.array(np.squeeze(on_object), dtype=np.int32))

                if self.data_save:
                    self.save_data(ind, pos)

            object_poses.append(pos)
            
        return object_poses, on_object_list

                 
    def save_data(self, ind, pose):

        for i in range(len(self.objects_BB)): 

            if ind == i :
                continue

            if self.is_overlapping(ind, i):
                return
        
        obj_class = self.objects_BB["name"][ind]
        bb_size = self.object_unique_size(ind, self.obj_conf[obj_class]['unique'])

        txt = obj_class + ".txt"
        input = str(bb_size) + " " + str(pose[2]) + "\n"
        with open( os.path.join( self.data_dir, txt ), 'a' ) as file:
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