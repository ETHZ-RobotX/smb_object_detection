import numpy as np
import sensor_msgs.point_cloud2 as pc2


def pointcloud2_to_xyzi(pointcloud2):
    pc_list = pc2.read_points_list(pointcloud2, skip_nans=True)
    xyzi = np.zeros((len(pc_list),4))
    for ind, p in enumerate(pc_list):
        xyzi[ind,0] = p[0]
        xyzi[ind,1] = p[1]
        xyzi[ind,2] = p[2]
        xyzi[ind,3] = p[3]
    return xyzi