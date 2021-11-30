import numpy as np
import cv2

class Object:
    def __init__(self, name_space, id, sample_count):
        self.name_space     = name_space
        self.id             = id
        self.sample_count   = sample_count

        self.pos_samples    = np.empty((0,3))
        self.dist_samples   = np.empty(0) 
        self.size_samples   = np.empty(0)
        self.gt_pos         = np.empty((0,3))

    def add_sample(self, pos_sample, relative_pos = None, size_sample = None):

        if self.pos_samples.shape[0] < self.sample_count :
            self.pos_samples = np.vstack((self.pos_samples, pos_sample))
            
            if relative_pos is not None:
                # TODO: Add distance/size relationship
                print("!!!DISTANCE/SIZE RELATION WAS NOT IMPLEMENTED!!!")

        else:
            self.pos_samples = np.vstack((self.pos_samples[1:,:], pos_sample))
            
            if relative_pos is not None:
                # TODO: Add distance/size relationship
                print("!!!DISTANCE/SIZE RELATION WAS NOT IMPLEMENTED!!!")  

    def get_pose(self):
        return np.mean(self.pos_samples, axis=0)
    
    def update_gt_pos(self, new_gt_pos):
        self.gt_pos = new_gt_pos
    
    def get_pos_error(self):
        #TODO: Add error calculation
        print("!!! POS ERROR CALCULATION WAS NOT IMPLEMENTED !!!")
    
    def get_pos_stats(self):
        stats = {}
        stats['mean']   = np.mean   (self.pos_samples, axis=0) 
        stats['median'] = np.median (self.pos_samples, axis=0)
        stats['std']    = np.std    (self.pos_samples, axis=0) 
        
        return stats
        
        