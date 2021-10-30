#!/usr/bin/env python
from ros_numpy import point_cloud2
import rospy
import ros_numpy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import tf




from object_detection.objectdetector import ObjectDetector
from object_detection.imagehandler import ImageHandler

class Node:
    def __init__(self):
        # Node related
        # TODO: Add self.lidar_topic
        self.camera_topic                   = rospy.get_param('camera_topic', '/versavis/cam0/image_raw')
        self.lidar_topic                    = rospy.get_param('lidar_topic', '/rslidar_points')
        self.node_name                      = rospy.get_param('node_name', 'listener')

        rospy.init_node(self.node_name, anonymous=True)

        # TODO: Add self.lidar_sub
        self.camera_sub                     = message_filters.Subscriber(self.camera_topic, Image)
        self.cv_bridge                      = CvBridge()
        self.lidar_sub                      = message_filters.Subscriber(self.lidar_topic, PointCloud2)

        # TODO: Add self.lidar_sub
        self.synchronizer                   = message_filters.ApproximateTimeSynchronizer([self.camera_sub, self.lidar_sub], 1, 0.05, reset=True)

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


        ##  rosrun tf tf_echo "rslidar" "blackfly_right_optical_link" 1
        ## - Translation: [-0.045, -0.293, -0.241]
        ## - Rotation: in Quaternion [0.000, -0.707, 0.707, -0.000]
        ##             in RPY (radian) [-1.570, -0.000, -3.141]
        ##             in RPY (degree) [-89.954, -0.000, -179.954]

        R_camera_lidar = np.float64([ [-1.0000000, 0.0000000,  0.0000000],
                                        [0.0000000, 0.0000000, -1.0000000],
                                        [0.0000000, -1.0000000,  0.0000000] ])

        t_camera_lidar = np.float64([-0.045, -0.293, -0.241])

        R = R_camera_lidar
        t = t_camera_lidar

        # -------- TODO: REMOVE HARDCODED PARAMS ---------------------------

        self.imagehandler.set_cameraparams(K, dist, wh)
        self.imagehandler.set_transformationparams(R,t)
        
        self.detector                       = ObjectDetector(self.detector_cfg)
        print("Detector is set")


    def run(self):
        def callback(image_msg, lidar_msg):
            if image_msg.height != 0:
                cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

                scale_percent = 50 # percent of original size
                width = int(cv_image.shape[1] * scale_percent / 100)
                height = int(cv_image.shape[0] * scale_percent / 100)
                dim = (width, height)

                # undistorted_cv_image = cv_image
                undistorted_cv_image = self.imagehandler.undistort(cv_image)

                full = np.hstack((cv2.resize(cv_image, dim, interpolation = cv2.INTER_AREA),cv2.resize(undistorted_cv_image, dim, interpolation = cv2.INTER_AREA)))
                
                cv2.imshow("result", full)
                cv2.waitKey(1)

                #point_cloud = np.float32(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar_msg))
                #
                #img_pts = self.imagehandler.projectPoints(point_cloud)
                ## img_pts_cv, img_pts_classic = self.imagehandler.projectPoints2(point_cloud)
                ## print(img_pts)
                #                
                #result = self.detector.detect(undistorted_cv_image)

                #for pt in img_pts:
                #    cv2.circle(result, tuple(pt), 1, (255,0,0))
                #
                #cv2.imshow("result", result)
                #cv2.waitKey(1)

        self.synchronizer.registerCallback(callback)
        rospy.spin()

    def run2(self):
        def callback(image_msg, lidar_msg):
            # If Image Message is not empty 
            if image_msg.height != 0:               
                
                # transform the image msg to numpy array after encoding
                cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

                # undistorted_cv_image = cv_image
                undistorted_cv_image = self.imagehandler.undistort(cv_image)

                # transform the pointcloud msg to numpy array and remove nans
                point_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(lidar_msg)
                mask = np.isfinite(point_cloud['x']) & np.isfinite(point_cloud['y']) & np.isfinite(point_cloud['z'])
                point_cloud = point_cloud[mask]

                # project points onto image
                point_cloud_XYZ = ros_numpy.point_cloud2.get_xyz_points(point_cloud, remove_nans=False)
                img_pts, indices = self.imagehandler.projectPoints(point_cloud_XYZ)
                
                # Detect objects 
                result = self.detector.detect(undistorted_cv_image, return_image=True)


                #for pt in img_pts:
                #    cv2.circle(result, tuple(pt), 1, (255,0,0))
                #
                #cv2.imshow("result", result)
                #cv2.waitKey(1)

        self.synchronizer.registerCallback(callback)
        rospy.spin()


if __name__ == '__main__':

    node = Node()
    print("Detection started")
    node.run2()
    
