from PIL.Image import FASTOCTREE
import numpy as np
from datetime import datetime
import os

AXIS = {'X' : 0,
        'Y' : 1,
        'Z' : 2,
        '-X' : -0,
        '-Y' : -1,
        '-Z' : -2}



class Map:
    def __init__(self, up, path, time, ceiling_th = 3, floor_th = 0.5, map_2D=None):
        """
        Map Object init function

        Args: 
            up : string, up axis of the map. Possible values (X,Y,Z,-X,-Y,-Z)
            ceiling_th : float, ceiling threshold, Up values higher than this threshold will be discarded
            floot_th : float, floor threhold, Up values lower than this threshold will be discarded
            map_2D : numpy Nx2 array. 2D map that contains only X, Y points  

        """
        
        self.up         = np.abs(AXIS[up])
        self.up_sign    = np.sign(AXIS[up]) 
        self.map_2D     = map_2D

        self.ceiling_th = ceiling_th
        self.floor_th   = floor_th

        self.april_ids  = ["0", "1", "2", "3", "4", \
                           "5", "6", "7", "8", "9", \
                           "14", "17"]

        self.now  = time
        self.file_name = path

        self.april_pos   = {}
        self.april_m_pos = {}
        self.april_color = {}        

        for id in self.april_ids:
            self.april_pos[id]      = np.empty((0,3))
            self.april_m_pos[id]    = None
            self.april_color[id]    = np.random.rand(3)
            

    def buildMap(self, points):
        if self.map_2D is None:
            # TODO: Add negative up sign
            remaining_idx = np.logical_and((points[:,self.up] > self.floor_th), (points[:,self.up] < self.ceiling_th))

            walls = points[remaining_idx]
            
        # TODO: Add given map

        # TODO: Remove close points 
        # self.map_2D = self.to2D(walls)
        self.map_2D = walls

    def to2D(self, points):
        points[:, self.up] = np.zeros((len(points[:, self.up])))
        return points


    def seeTag(self, id, pose):
        self.april_pos[str(id)] = np.vstack([self.april_pos[str(id)], pose])
        self.april_m_pos[str(id)] = np.mean(self.april_pos[str(id)], axis=0)

    
    def dumpStats(self):
       
        now  = datetime.now()
        now  = now.strftime("%H_%M_%S")

        with open(self.file_name, "w+") as f:

            f.write("Statistic of Localization of April Tags \n")
            f.write("Date and Time: " + self.now + "\n")
            f.write("\n")

            for id in self.april_ids:
                f.write("Tag " + str(id) + ", color: "+ str(self.april_color[str(id)]) + "\n")
                f.write("Mean: " + str(np.mean(self.april_pos[str(id)], axis=0)) + "\n")
                f.write("Median: " + str(np.median(self.april_pos[str(id)], axis=0)) + "\n")
                f.write("Std: " + str(np.std(self.april_pos[str(id)], axis=0)) + "\n")
                f.write("------------------------------------------------------------- \n")



            f.write("Last Update at " + now + "\n")