#!/usr/bin/env python
import cv2
import rospy
import tf2_ros
import ros_numpy
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from object_visualization.msg import ObjectDetection, ObjectDetectionArray

from object_visualization.utils import *

class Node:
    def __init__(self):

        rospy.init_node("visualize", anonymous=True)

        self.object_pub_topic               = rospy.get_param('~object_topic', '/objects')
        self.image_pub_topic                = rospy.get_param('~out_image_pub_topic', "/versavis/cam0/objects")
        self.visualize_all                  = rospy.get_param('~visualize_all', False)
        self.map_frame                      = rospy.get_param('~map_frame', 'map')

        self.tf_buffer                      = tf2_ros.Buffer(rospy.Duration(4.0)) #tf buffer length
        self.tf_listener                    = tf2_ros.TransformListener(self.tf_buffer)
        self.TF_br                          = tf2_ros.StaticTransformBroadcaster()
        self.cv_bridge                      = CvBridge()

        self.marker_pub                     = rospy.Publisher(self.object_pub_topic , MarkerArray, queue_size=10)
        self.image_pub                  = rospy.Publisher(self.image_pub_topic , Image, queue_size=5)

        self.obj_id                         = 0
        self.marker_color                   = {}

        if self.objectdetector_cfg['classes'] is not None:
            for c in self.objectdetector_cfg['classes']:
                self.marker_color [c] = np.random.rand(3)
    
    def object_callback(self, detection):
        img = self.cv_bridge.imgmsg_to_cv2(detection.detections_image , "bgr8")
        point_cloud_2D = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(detection.pointcloud_in_frame_2D)
        point_cloud_3D = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(detection.pointcloud_in_frame_3D)
        objects = detection.detections

        if self.visualize_all:
            for idx, pt in enumerate(point_cloud_2D): 
                dist = point_cloud_3D[idx, 2]
                color = depth_color(dist, min_d=0.2, max_d=20)

                try:
                    cv2.circle(img, pt[:2], 1, color)
                except:
                    print("Cannot Circle")
        else:
            for idx, object in enumerate(objects):
                on_obj_indices = object.on_object_point_indices
                obj_class = object.class_id
                for idx, pt in enumerate(point_cloud_2D[on_obj_indices]): 
                    dist = point_cloud_3D[idx, 2]
                    try:
                        cv2.circle(img, pt[:2], 1, CLASS_COLOR[obj_class])
                    except:
                        print("Cannot Circle")
        
        img_msg = self.cv_bridge.cv2_to_imgmsg(img, 'bgr8')
        img_msg.header.frame_id = detection.header.frame_id
        self.image_pub.publish(img_msg)

        markers = MarkerArray()
        for idx, object in enumerate(objects):
            obj_class = object.class_id
            obj_id = object.id

            xyz = [object.pose.x, object.pose.y, object.pose.z]

            self.TF_br.sendTransform(transformstamped_(detection.header.frame_id, obj_class + str(obj_id), \
                                                       detection.header.stamp, xyz, [0,0,0,1]))
            
            try:
                transform = self.tf_buffer.lookup_transform_full(self.map_frame, detection.header.stamp, \
                                                                obj_class + str(obj_id), detection.header.stamp, \
                                                                self.map_frame, rospy.Duration(2.0))
            except:
                print("cannot transform")
                continue

            obj_in_map = np.array([transform.transform.translation.x, \
                                    transform.transform.translation.y, \
                                    transform.transform.translation.z ])

            markers.markers.append(marker_(obj_class + str(obj_id), obj_id, obj_in_map, detection.header.stamp, CLASS_COLOR[obj_class]/255.0, self.map_frame))

        self.marker_pub.publish(markers)  


    def run(self):
        self.tag_sub    = rospy.Subscriber(self.object_topic, ObjectDetectionArray, self.object_callback)

        rospy.spin()

if __name__ == '__main__':

    node = Node()
    node.run()
