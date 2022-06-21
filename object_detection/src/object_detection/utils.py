import numpy as np
import sensor_msgs.point_cloud2 as pc2

from geometry_msgs.msg import Pose, TransformStamped
from visualization_msgs.msg import Marker

Z_UPWARDS   = 2

def filter_ground(point3D, ground_percentage, upward = Z_UPWARDS):
    """ Filter the ground according to given percentage. 
    Args:
        point3D                   : 3D point translated point cloud 

    Returns:
        non_groun_point3D         : indices of points that are not ground
    """

    if upward < 0:
        indices = np.nonzero(point3D[:,-upward] < max(point3D[:,-upward])*(100-ground_percentage)/100)[0]
        
    else:
        indices = np.nonzero(point3D[:,upward] > min(point3D[:,upward])*(100-ground_percentage)/100)[0]
    
    return point3D[indices]

def pointcloud2_to_xyzi(pointcloud2):
    """ pointcloud2 to xyzi numpu array transformation
    Args:
        pointcloud2  : pointcloud2 ros message

    Returns:
        xyzi         : numpy array of nx4 of coordinate of lidar points 
    """
    pc_list = pc2.read_points_list(pointcloud2, skip_nans=True)
    xyzi = np.zeros((len(pc_list),4))
    for ind, p in enumerate(pc_list):
        xyzi[ind,0] = p[0]
        xyzi[ind,1] = p[1]
        xyzi[ind,2] = p[2]
        xyzi[ind,3] = p[3]
    return xyzi

def check_validity_image_info(K, w, h):
    l = [K,w,h]
    if all( x is not None for x in l ) and K[2,2] > 0 and w > 0 and h > 0 :
        return True
    else:
        return False

def check_validity_lidar2camera_transformation(R,t):
    l = [R,t]
    if all( x is not None for x in l ):
        return True
    else:
        return False

# ----------------------- Visualisation -----------------------------------
NO_POSE = -1

CLASS_COLOR = { 'person'  : (255,0,0),
                'bicycle' : (0,255,0),
                'umbrella': (0,0,255),
                'bottle' : (125,125,0),
                'clock' : (0,125,125),
                'bench' : (125,0,125)}

def marker_(ns, marker_id, pos, stamp ,color, frame_id = "map"):
    
    marker = Marker()
    marker.ns = str(ns)
    marker.header.frame_id = frame_id
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

    marker.id = marker_id
    
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 1.0


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