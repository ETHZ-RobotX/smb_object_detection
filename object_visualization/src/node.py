#!/usr/bin/env python
import rospy
import tf2_ros
import ros_numpy
import numpy as np
from cv_bridge import CvBridge

from geometry_msgs.msg import Pose, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from object_detection.msg import ObjectDetection, ObjectDetectionArray



class Node:
    def __init__(self):

        rospy.init_node("visualize", anonymous=True)

        self.object_topic                   = rospy.get_param('~object_topic', '/obejcts')
        self.visualize_all                  = rospy.get_param('~visualize_all', False)

        self.tf_buffer                      = tf2_ros.Buffer(rospy.Duration(4.0)) #tf buffer length
        self.tf_listener                    = tf2_ros.TransformListener(self.tf_buffer)
        self.TF_br                          = tf2_ros.StaticTransformBroadcaster()
        self.cv_bridge                      = CvBridge()

        self.marker_pub                     = rospy.Publisher(self.obj_marker_topic , MarkerArray, queue_size=10)
        self.obj_id                         = 0
        self.marker_color                   = {}



        if self.objectdetector_cfg['classes'] is not None:
            for c in self.objectdetector_cfg['classes']:
                self.marker_color [c] = np.random.rand(3)
    
    def object_callback(self, objects):
        img = self.cv_bridge.imgmsg_to_cv2(objects.detections_image , "bgr8")
        point_cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(objects.pointcloud_in_frame_2D)

        if self.visualize_all:
            a = 0
        else:
            b = 0


    def run(self):
        self.tag_sub    = rospy.Subscriber(self.object_topic, ObjectDetectionArray, self.object_callback)

        rospy.spin()

if __name__ == '__main__':

    node = Node()
    node.run()
