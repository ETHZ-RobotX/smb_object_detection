#!/usr/bin/env python
from numpy.core.fromnumeric import transpose
from ros_numpy import point_cloud2
from rospy import exceptions
import sensor_msgs 
import rospy
from roslib import message
import ros_numpy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import tf
import tf2_ros
import tf2_geometry_msgs
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose
import struct

from map.map import Map
from apriltag_ros.msg import *

def marker_(id, pos, color, type="all"):

    marker = Marker()
    marker.ns = str(id)
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

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 1.0
    else:
        
        marker.id = np.random.default_rng().integers(0,10000)
        
        marker.scale.x = 0.1
        marker.scale.y = 0.1
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


IDs = [0,1,2,3,4,5,6,7,8,9,14,17]

class Node:
    def __init__(self):
        # Node related
        self.pc_map_topic                   = rospy.get_param('pc_map_topic', '/icp_node/icp_map')
        self.lidar_topic                    = rospy.get_param('lidar_topic', '/rslidar_points')
        self.tag_topic                      = rospy.get_param('tag_topic', '/tag_detections')
        self.node_name                      = rospy.get_param('node_name', 'map')

        rospy.init_node(self.node_name, anonymous=True)
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0)) #tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.mapper = Map("Z")

        # Publisher
        self.pc_map_pub                     = rospy.Publisher(self.pc_map_topic + '_2D', PointCloud2, queue_size=1)
        self.lidar_pub                      = rospy.Publisher(self.lidar_topic + '_2D', PointCloud2, queue_size=10)
        self.marker_pub                     = rospy.Publisher(self.tag_topic + '_marker', MarkerArray, queue_size=10)

    def pc_map_callback(self, data):
        # if self.mapper.map_2D is None:
        point_cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
        print(point_cloud.shape)
        self.mapper.buildMap(point_cloud)

        pts = np.zeros((self.mapper.map_2D.shape[0], 7))
        pts[:,:3] = self.mapper.map_2D

        msg = point_cloud_(pts,'map')
        while True:
            self.pc_map_pub.publish(msg)
            rospy.sleep(10)
            

    def tag_callback(self, data):
        
        markers = MarkerArray()

        for tag in data.detections:
            # print("Tag ",tag.id[0]," has been seen." )


            transform = self.tf_buffer.lookup_transform('map',
                                        'tag_' + str(tag.id[0]), #source frame
                                        rospy.Time(0),
                                        rospy.Duration(2.0)) #get the tf at first available time) #wait for 1 second

            xyz = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z ])
            # xyz = np.array([tag.pose.pose.pose.position.x, tag.pose.pose.pose.position.y, tag.pose.pose.pose.position.z ])    
            self.mapper.seeTag(tag.id[0], xyz)

            color = self.mapper.april_color[str(tag.id[0])]
            m_pos = self.mapper.april_m_pos[str(tag.id[0])]
            pos   = self.mapper.april_pos[str(tag.id[0])]

            for p in pos:
                markers.markers.append(marker_(str(tag.id[0]), p, color))

            markers.markers.append(marker_(str(tag.id[0]), m_pos, color, type="mean"))
            self.marker_pub.publish(markers)


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

    
    def run(self):
        # self.pc_map_sub                     = rospy.Subscriber(self.pc_map_topic, PointCloud2, self.pc_map_callback)
        # self.lidar_sub                      = rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback)
        self.tag_sub                        = rospy.Subscriber(self.tag_topic , AprilTagDetectionArray, self.tag_callback) 

        rospy.spin()

if __name__ == '__main__':

        node = Node()
        print("Mapping started")
        node.run()


    
