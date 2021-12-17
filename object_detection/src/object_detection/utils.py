import numpy as np

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, TransformStamped

"""
def dumpStats():
    
    print("Mean tag pose: " + str(self.mean_tag_pos) + "\n")

    self.pos_error = np.array(self.pos_error)

    now  = datetime.now()
    now  = now.strftime("%H_%M_%S")

    with open("/home/oilter/Documents/Statistics/" + self.statistic_txt + ".txt", "w") as record:
        record.write("Statistic of Localization of Objects \n")
        record.write("Date and Time: " + now + "\n")
        record.write("\n")

        record.write("!!!!!! \n")
        record.write("Discarding errors more than 2 meters while calculating stats\n")
        record.write("!!!!!! \n")

        error_2     = self.pos_error < 2.0

        record.write("Pos Error \n")
        record.write("Mean: " + str(np.mean(self.pos_error[error_2])) + "\n")
        record.write("Median: " + str(np.median(self.pos_error[error_2])) + "\n")
        record.write("Std: " + str(np.std(self.pos_error[error_2])) + "\n")
        record.write("Max Error in 2 m: " + str(max(self.pos_error[error_2])) + "\n")
        record.write("Max Error in whole: " + str(max(self.pos_error)) + "\n")
        
        
        record.write("\n")
        record.write("Number of whole samples: " + str(len(self.pos_error)) + "\n")
        
        error_2     = self.pos_error < 2.0
        error_15     = self.pos_error < 1.5
        error_1     = self.pos_error < 1.0
        error_05    = self.pos_error < 0.5
        error_025   = self.pos_error < 0.25
        error_01   = self.pos_error < 0.1

        record.write("Number of samples that has error less than 2.0 m: " + str(len(self.pos_error[error_2])) + "\n")
        record.write("Number of samples that has error less than 1.5 m: " + str(len(self.pos_error[error_15])) + "\n")
        record.write("Number of samples that has error less than 1.0 m: " + str(len(self.pos_error[error_1])) + "\n")
        record.write("Number of samples that has error less than 0.5 m: " + str(len(self.pos_error[error_05])) + "\n")
        record.write("Number of samples that has error less than 0.25 m: " + str(len(self.pos_error[error_025])) + "\n")
        record.write("Number of samples that has error less than 0.1 m: " + str(len(self.pos_error[error_01])) + "\n")

        record.write("Mean tag pose: " + str(self.mean_tag_pos) + "\n")

        record.write("------------------------------------------------------------- \n")

    print("Statistic of Localization of Objects \n")
    print("Date and Time: " + now + "\n")
    print("\n")

    print("!!!!!! \n")
    print("Discarding errors more than 2 meters while calculating stats\n")
    print("!!!!!! \n")

    print("Pos Error \n")
    print("Mean: " + str(np.mean(self.pos_error[error_2])) + "\n")
    print("Median: " + str(np.median(self.pos_error[error_2])) + "\n")
    print("Std: " + str(np.std(self.pos_error[error_2])) + "\n")
    print("Max Error in 2 m: " + str(max(self.pos_error[error_2])) + "\n")
    print("Max Error in whole: " + str(max(self.pos_error)) + "\n")
            
    print("\n")
    print("Number of whole samples: " + str(len(self.pos_error)) + "\n")

    print("Number of samples that has error less than 2.0 m: " + str(len(self.pos_error[error_2])) + "\n")
    print("Number of samples that has error less than 1.5 m: " + str(len(self.pos_error[error_15])) + "\n")
    print("Number of samples that has error less than 1.0 m: " + str(len(self.pos_error[error_1])) + "\n")
    print("Number of samples that has error less than 0.5 m: " + str(len(self.pos_error[error_05])) + "\n")
    print("Number of samples that has error less than 0.25 m: " + str(len(self.pos_error[error_025])) + "\n")
    print("Number of samples that has error less than 0.1 m: " + str(len(self.pos_error[error_01])) + "\n")

    print("Mean tag pose: " + str(self.mean_tag_pos) + "\n")

    print("------------------------------------------------------------- \n")
"""
def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 50.0
    h60f = np.floor(h60)
    hi = int(h60f) % 5
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def depth_color(val, min_d=0.2, max_d=20):
    """
    print Color(HSV's H value) corresponding to distance(m)
    close distance = red , far distance = blue
    """
    # np.clip(val, 0, max_d, out=val)

    hsv = (-1*((val - min_d) / (max_d - min_d)) * 255).astype(np.uint8)
    return hsv2rgb(hsv,1,1)

def marker_(ns, marker_id, pos, stamp ,color, type="all"):

    marker = Marker()
    marker.ns = str(ns)
    marker.header.frame_id = "map"
    marker.header.stamp = stamp
    marker.type = 2
    marker.action = 0
    marker.pose = Pose()

    marker.pose.position.x = pos[0]
    marker.pose.position.y = pos[1]
    marker.pose.position.z = pos[2]

    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0

    if type=="mean":
        marker.id = 10000

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 1.0
    else:
        
        marker.id = marker_id
        
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.3


    marker.frame_locked = False
        
    return marker

def transformstamped_(frame_id, child_id, time, pose, rot):
    t = TransformStamped()

    t.header.stamp = time
    t.header.frame_id = frame_id
    t.child_frame_id = child_id
    t.transform.translation.x = pose[0]
    t.transform.translation.y = pose[1]
    t.transform.translation.z = pose[2]
    t.transform.rotation.x = rot[0]
    t.transform.rotation.y = rot[1]
    t.transform.rotation.z = rot[2]
    t.transform.rotation.w = rot[3]

    return t

def pointcloud_filter_ground(xyz):
    min_z = min(xyz[:,2])

    non_ground_indices = np.argwhere(xyz[:,2] > min_z*0.85).flatten()
    return(xyz[non_ground_indices])

def param_array_parser(param_array):
    if param_array != 'None':
        classes = np.array([[int(x.strip(' ')) for x in ss.lstrip(' [,').split(', ')] for ss in param_array.rstrip(']').split(']')])
        return list(classes.flatten())
    else:
        return None