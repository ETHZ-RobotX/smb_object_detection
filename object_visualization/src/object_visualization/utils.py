import numpy as np

from geometry_msgs.msg import Pose, TransformStamped
from visualization_msgs.msg import Marker

NO_POSE = -1

CLASS_COLOR = {'person'  : (255,0,0),
               'bicycle' : (0,255,0),
               'umbrella': (0,0,255),
               'bottle' : (125,125,0),
               'clock' : (0,125,125)}

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