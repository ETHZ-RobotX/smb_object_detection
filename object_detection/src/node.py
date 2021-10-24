#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2

from object_detection.detector import Detector

class Node:
    def __init__(self):
        # Node related
        # TODO: Add self.lidar_topic
        self.camera_topic                   = rospy.get_param('camera_topic', '/versavis/cam0/image_raw')
        # self.lidar_topic                   = rospy.get_param('lidar_topic', '/webcam/image_raw')
        self.node_name                      = rospy.get_param('node_name', 'listener')

        rospy.init_node(self.node_name, anonymous=True)

        # TODO: Add self.lidar_sub
        self.camera_sub                     = message_filters.Subscriber(self.camera_topic, Image)
        self.cv_bridge                      = CvBridge()
        # self.lidar_sub                      = message_filters.Subscriber(self.lidar_topic, LaserScan)

        # TODO: Add self.lidar_sub
        self.synchronizer                   = message_filters.TimeSynchronizer([self.camera_sub], 1)

        # Detector related
        self.detector_cfg = {}
        self.detector_cfg['architecture']   = rospy.get_param('architecture', 'yolo')
        # self.detector_cfg['architecture']   = rospy.get_param('architecture', 'detectron')
        self.detector_cfg['model']          = rospy.get_param('model', 'yolov5n')
        # self.detector_cfg['model']          = rospy.get_param('model', 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
        self.detector_cfg['checkpoint']     = rospy.get_param('checkpoint', None)
        self.detector_cfg['device']         = rospy.get_param('device', 'cpu')
        self.detector_cfg['confident']      = rospy.get_param('confident', 0.5)
        self.detector_cfg['iou']            = rospy.get_param('iou', 0.45)
        self.detector_cfg['classes']        = rospy.get_param('classes',  None)

        self.detector                       = Detector(self.detector_cfg)
        print("Detector is set")
        

    def run(self):
        def callback(image_msg):
            if image_msg.height != 0:
                cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg)
                result = self.detector.detect(cv_image)
                cv2.imshow("result", result)
                cv2.waitKey(1)
        self.synchronizer.registerCallback(callback)
        rospy.spin()


if __name__ == '__main__':

    node = Node()
    print("Detection started")
    node.run()
    
