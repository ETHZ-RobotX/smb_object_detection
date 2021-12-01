#!/usr/bin/env python
import rospy
import ros_numpy
import numpy as np
from sensor_msgs.msg import PointCloud2
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose
import rosbag
from datetime import datetime
import os

from map.map import Map
from apriltag_ros.msg import *


import warnings
warnings.filterwarnings("ignore")

def marker_(ns, marker_id, pos, color, type="all"):

    marker = Marker()
    marker.ns = str(ns)
    marker.header.frame_id = "map"
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

        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 1.0
    else:
        
        marker.id = marker_id
        
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.3


    marker.frame_locked = False
        
    return marker

def point_cloud_(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzrgba')]

    header = Header(seq=0,frame_id=parent_frame, stamp=rospy.Time.now())

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 7),
        row_step=(itemsize * 7 * points.shape[0]),
        data=data
    )

def tags(tag, tf_buffer):
    tag_frame_name = "tag_" + str(tag)

    try:
        transform = tf_buffer.lookup_transform('map',
                    tag_frame_name, #source frame
                        rospy.Time(0),
                        rospy.Duration(0.001)) #get the tf at first available time) #wait for 1 second
    except:
        print(tag_frame_name, " is not in fov")
    # print(transform)

IDs = [0,1,2,3,4,5,6,7,8,9,13,14,17]

class Node:
    def __init__(self):
        rospy.init_node("mapper", anonymous=True)
        # Node related
        self.pc_map_topic                   = rospy.get_param('~pc_map_topic', '/icp_node/icp_map')
        self.lidar_topic                    = rospy.get_param('~lidar_topic', '/rslidar_points')
        self.tag_topic                      = rospy.get_param('~tag_topic', '/tag_detections')
        self.output_dir                     = rospy.get_param('~output_dir', '.')
        self.log_period                     = rospy.get_param('~log_period', 20)
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0)) #tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        print(self.log_period)

        now  = datetime.now()
        now  = now.strftime("%m_%d_%Y_%H_%M")
        file_dir = os.path.join(self.output_dir, "mapping_stats_" + now)

        self.mapper = Map("Z", file_dir + ".txt", now )
        if self.log_period > 0:
            self.bag = rosbag.Bag(file_dir+'.bag', 'w')

        # Publisher
        self.pc_map_pub                     = rospy.Publisher(self.pc_map_topic + '_2D', PointCloud2, queue_size=1)
        self.lidar_pub                      = rospy.Publisher(self.lidar_topic + '_2D', PointCloud2, queue_size=10)
        self.marker_pub                     = rospy.Publisher(self.tag_topic + '_marker', MarkerArray, queue_size=10)

        

    def pc_map_callback(self, data):
        # if self.mapper.map_2D is None:
        if self.log_period > 0:
            self.bag.write('/icp_node/icp_map', data)
        point_cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
        self.mapper.buildMap(point_cloud)

        pts = np.zeros((self.mapper.map_2D.shape[0], 7))
        pts[:,:3] = self.mapper.map_2D

        msg = point_cloud_(pts,'map')
        if self.log_period > 0:
            self.bag.write('/icp_node/icp_map_2D', msg)
        while True:
            self.pc_map_pub.publish(msg)
            if self.log_period > 0:
                self.bag.write('/icp_node/icp_map', data)
                self.bag.write('/icp_node/icp_map_2D', msg)
            rospy.sleep(self.log_period)
            

    def tag_callback(self, data):
        
        markers = MarkerArray()

        for tag in data.detections:
            # print("Tag ",tag.id[0]," has been seen." )


            transform = self.tf_buffer.lookup_transform('map',
                                        'tag_' + str(tag.id[0]), #source frame
                                        rospy.Time(0),
                                        rospy.Duration(5.0)) #get the tf at first available time) #wait for 5 second

            xyz = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z ])
            # xyz = np.array([tag.pose.pose.pose.position.x, tag.pose.pose.pose.position.y, tag.pose.pose.pose.position.z ])    
            self.mapper.seeTag(tag.id[0], xyz)

            color = self.mapper.april_color[str(tag.id[0])]
            m_pos = self.mapper.april_m_pos[str(tag.id[0])]
            pos   = self.mapper.april_pos[str(tag.id[0])]

            for id,p in enumerate(pos):
                markers.markers.append(marker_(str(tag.id[0]), id, p, color))

            markers.markers.append(marker_(str(tag.id[0]),id, m_pos, color, type="mean"))
            self.marker_pub.publish(markers)

        if self.log_period > 0 and int(rospy.get_time()) % self.log_period == 0 and int(rospy.get_time()) != 0:
            self.save_stats()
            


    def lidar_callback(self, data):

        transform = self.tf_buffer.lookup_transform('map',
                                       data.header.frame_id, #source frame
                                       rospy.Time(0),
                                       rospy.Duration(1.0)) #get the tf at first available time) #wait for 1 second

        # print(transform)
        cloud_out = do_transform_cloud(data, transform)

        point_cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud_out)
        lidar_2D = self.mapper.to2D(point_cloud)

        pts = np.ones((lidar_2D.shape[0], 7))
        pts[:,:3] = lidar_2D
        msg = point_cloud_(pts,'map')

        for id in IDs:
            tags(id, self.tf_buffer)

        self.lidar_pub.publish(msg)

    def save_stats(self):
        self.mapper.dumpStats()

        markers = MarkerArray()
        for id in IDs:
            color = self.mapper.april_color[str(id)]
            m_pos = self.mapper.april_m_pos[str(id)]
    
            if m_pos is None:
                continue

            pos   = self.mapper.april_pos[str(id)]

            for i,p in enumerate(pos):
                markers.markers.append(marker_(str(id),i, p, color))

            markers.markers.append(marker_(str(id),i, m_pos, color, type="mean"))
            self.marker_pub.publish(markers)
            self.bag.write('/tags', markers)


    
    def run(self):
        self.pc_map_sub                     = rospy.Subscriber(self.pc_map_topic, PointCloud2, self.pc_map_callback)
        self.tag_sub                        = rospy.Subscriber(self.tag_topic , AprilTagDetectionArray, self.tag_callback) 

        rospy.spin()

if __name__ == '__main__':

        node = Node()
        print("Mapping started")
        node.run()
        if node.log_period > 0:
            node.save_stats()
            node.bag.close()
        print("End of Mapping")
        
    
