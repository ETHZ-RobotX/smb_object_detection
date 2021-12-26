import os
import numpy as np
import yaml

# Save height - distance, area - distance, length - distance pair for every object in X Y format
# Ex: For bike height - distance is a better pair 
# At every detection cycle add more data 
# When the program ends fit RANSAC y = ax + b 
# Delete all points create points on y = ax + b only! Think how many points 

# Dist estimation from size + lidar --> KALMAN ? 


class ObjectLerner:
    def __init__(self, data):

        self.data_dir = data

        with open(data) as file:
            self.config             = yaml.load(file, Loader=yaml.FullLoader)
            self.obj_conf           = self.config['objects']


    def set_data(self, class_id, bb_size, distance):
        
        txt = class_id + ".txt"
        input = str(bb_size) + " " + str(distance) + "\n"
        with open( os.path.join( self.data_dir, txt ), 'a' ) as file:
            file.write(input)

    def calculate_equations(self):
        a = 0