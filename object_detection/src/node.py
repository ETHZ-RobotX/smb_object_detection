#!/usr/bin/env python
import rospy
import ros_numpy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2


from object_detection.objectdetector import ObjectDetector
from object_detection.imagehandler import ImageHandler

class Node:
    def __init__(self):
        # Node related
        # TODO: Add self.lidar_topic
        self.camera_topic                   = rospy.get_param('camera_topic', '/versavis/cam0/image_raw')
        self.lidar_topic                   = rospy.get_param('lidar_topic', '/rslidar_points')
        self.node_name                      = rospy.get_param('node_name', 'listener')

        rospy.init_node(self.node_name, anonymous=True)

        # TODO: Add self.lidar_sub
        self.camera_sub                     = message_filters.Subscriber(self.camera_topic, Image)
        self.cv_bridge                      = CvBridge()
        self.lidar_sub                      = message_filters.Subscriber(self.lidar_topic, PointCloud2)

        # TODO: Add self.lidar_sub
        self.synchronizer                   = message_filters.ApproximateTimeSynchronizer([self.camera_sub, self.lidar_sub], 1, 0.1, reset=True)

        # Detector related
        self.detector_cfg = {}
        self.detector_cfg['architecture']   = rospy.get_param('architecture', 'yolo')
        self.detector_cfg['model']          = rospy.get_param('model', 'yolov5n')
        self.detector_cfg['checkpoint']     = rospy.get_param('checkpoint', None)
        self.detector_cfg['device']         = rospy.get_param('device', 'cpu')
        self.detector_cfg['confident']      = rospy.get_param('confident', 0.5)
        self.detector_cfg['iou']            = rospy.get_param('iou', 0.45)
        self.detector_cfg['classes']        = rospy.get_param('classes',  None)

        # Camera Params
        self.imagehandler                   = ImageHandler()

        # -------- TODO: REMOVE HARDCODED PARAMS ---------------------------
        K = np.eye(3)
        K[0,0] = 644.1589408974335
        K[0,2] = 694.3102357386883
        K[1,1] = 643.8998733804048
        K[1,2] = 574.1681961598792

        dist = np.float32([ 0.005691383154435742, -0.0006697996624948808, -0.0031151487145129318, 0.002980432455329788])

        wh = [1440, 1080]

        R = np.float64([[0.0333,-0.0107,0.9993],
             [-0.9993,0.0120,0.0334],
             [-0.0123,-0.9998,-0.0103]])
        
        R = np.float64([[0.0,0,1.0],
             [-1.0,0.0,0.0],
             [0.0,-1.0,0.0]])
        
        a = np.float64([ [0.0000000,  1.0000000,  0.0000000],
                [-1.0000000,  0.0000000,  0.0000000],
                    [0.0000000,  0.0000000,  1.0000000] ])

        R = a @ R

        R_lidar_2_camera = np.float64([ [-1.0000000, 0.0000000,  0.0000000],
                        [0.0000000, -0.0000000, -1.0000000],
                        [0.0000000, -1.0000000,  0.0000000] ])
  
        r_base_2_camera = np.float64([ [0.0000000,  0.0000000,  1.0],
            [-1.0,  0.0000000,  0.0000000],
            [0.0000000, -1.0,  0.0000000] ])

        r_base_2_lidar = np.float64([[ -0.0000000, -1.0000000,  0.0000000],
            [1.0000000, -0.0000000,  0.0000000],
            [0.0000000,  0.0000000,  1.0000000] ])


        # R = np.transpose(r_base_2_lidar)@r_base_2_camera   

        # t = np.transpose( np.float64([0.3956,0.2128,-0.2866]) )
        t = np.float64([0.29297, -0.04542, -0.2415])
        # t = np.float64([0,0,0])
        


        # t = r_base_2_camera @ t
        # -------- TODO: REMOVE HARDCODED PARAMS ---------------------------

        self.imagehandler.set_cameraparams(K, dist, wh)
        self.imagehandler.set_transformationparams(R,t)
        
        self.detector                       = ObjectDetector(self.detector_cfg)
        print("Detector is set")


    def run(self):
        def callback(image_msg, lidar_msg):
            if image_msg.height != 0:
                cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
                cv_image = self.imagehandler.undistort(cv_image)

                point_cloud = np.float32(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar_msg))
                img_pts = self.imagehandler.projectPoints(point_cloud)
                                
                result = self.detector.detect(cv_image)
 
                result[img_pts[:,1], img_pts[:,0], :] = [0,0,255]
                cv2.imshow("result", result)
                cv2.waitKey(1)
        self.synchronizer.registerCallback(callback)
        rospy.spin()


if __name__ == '__main__':

    node = Node()
    print("Detection started")
    node.run()
    
