#!/usr/bin/env python
from ros_numpy import point_cloud2
import rospy
import ros_numpy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import cv2
import tf

from object_detection.objectdetector import ObjectDetector
from object_detection.reproject import ImageHandler
from object_detection.modelextractor import *


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


def depth_color(val, min_d=0, max_d=20):
    """
    print Color(HSV's H value) corresponding to distance(m)
    close distance = red , far distance = blue
    """
    # np.clip(val, 0, max_d, out=val)

    hsv = (-1*((val - min_d) / (max_d - min_d)) * 255).astype(np.uint8)
    hsv2rgb(hsv,1,1)
    return hsv2rgb(hsv,1,1)

class Node:
    def __init__(self):

        rospy.init_node("objectify", anonymous=True)

        # Node related
        self.camera_topic                   = rospy.get_param('~camera_topic', '/versavis/cam0/undistorted')
        self.camera_info_topic              = rospy.get_param('~camera_info_topic', '/versavis/cam0/camera_info')
        self.lidar_topic                    = rospy.get_param('~lidar_topic', '/rslidar_points')

        self.camera_info_callback_sleep     = rospy.get_param('~camera_info_callback_sleep', 20)
        
        self.camera_sub                     = message_filters.Subscriber(self.camera_topic, Image)
        self.lidar_sub                      = message_filters.Subscriber(self.lidar_topic, PointCloud2)

        self.cv_bridge                      = CvBridge()

        self.synchronizer                   = message_filters.ApproximateTimeSynchronizer([self.camera_sub, self.lidar_sub], 1, 0.05, reset=True)
        
        # Output related
        self.visualize                      = rospy.get_param('visualize', True)
        self.visualize_all_points           = rospy.get_param('~visualize_all_points', False)
        self.out_image_pub_topic            = rospy.get_param('~out_image_pub_topic', "/versavis/cam0/undistorted/objects")
        self.out_image_pub                  = rospy.Publisher(self.out_image_pub_topic , Image, queue_size=5)


        # Detector related
        self.detector_cfg = {}
        self.detector_cfg['architecture']   = rospy.get_param('~architecture', 'yolo')
        self.detector_cfg['model']          = rospy.get_param('~model', 'yolov5n')
        self.detector_cfg['checkpoint']     = rospy.get_param('~checkpoint', None)
        self.detector_cfg['device']         = rospy.get_param('~device', 'cpu')
        self.detector_cfg['confident']      = rospy.get_param('~confident', 0.5)
        self.detector_cfg['iou']            = rospy.get_param('~iou', 0.45)
        classes                             = rospy.get_param('~classes',  None)
        if classes != 'None':
            classes = np.array([[int(x.strip(' ')) for x in ss.lstrip(' [,').split(', ')] for ss in classes.rstrip(']').split(']')])
            self.detector_cfg['classes'] = list(classes.flatten())
        else:
            self.detector_cfg['classes'] = None

        # Camera Params
        self.imagehandler                   = ImageHandler()

        # -------- TODO: REMOVE HARDCODED PARAMS ---------------------------

        ##  rosrun tf tf_echo "rslidar" "blackfly_right_optical_link" 1
        ## - Translation: [-0.045, -0.293, -0.241]
        ## - Rotation: in Quaternion [0.000, -0.707, 0.707, -0.000]
        ##             in RPY (radian) [-1.570, -0.000, -3.141]
        ##             in RPY (degree) [-89.954, -0.000, -179.954]

        R_camera_lidar = np.float64([ [-1.0000000, 0.0000000,  0.0000000],
                                        [0.0000000, 0.0000000, -1.0000000],
                                        [0.0000000, -1.0000000,  0.0000000] ])

        t_camera_lidar = np.float64([-0.045, -0.293, -0.241])

        # -------- TODO: REMOVE HARDCODED PARAMS ---------------------------

        self.imagehandler.set_transformationparams(R_camera_lidar,t_camera_lidar)
        
        self.detector                       = ObjectDetector(self.detector_cfg)
        print("Detector is set")

    def image_info_callback(self, camera_info):
        h = camera_info.height
        w = camera_info.width
        K = np.array(camera_info.K, dtype=np.float64).reshape(3,3)

        self.imagehandler.set_cameraparams(K, [w,h])
        rospy.sleep(self.camera_info_callback_sleep)

    def run(self):
        def callback(image_msg, lidar_msg):
            # If Image Message is not empty 
            if image_msg.height != 0:               
                # transform the image msg to numpy array
                cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg)

                # transform the pointcloud msg to numpy array and remove nans
                point_cloud_XYZ = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar_msg)
                point_cloud_XYZ = self.imagehandler.translatePoints(point_cloud_XYZ)
                img_pts, indices = self.imagehandler.projectPointsOnImage(point_cloud_XYZ)
                
                point_cloud_XYZ = point_cloud_XYZ[indices]

                # Detect objects 
                result = self.detector.detect(cv_image, return_image=True)

                if self.visualize and self.visualize_all_points:
                    for idx, pt in enumerate(img_pts):
                        dist = np.linalg.norm(point_cloud_XYZ[idx])
                        color = depth_color(dist)
                        cv2.circle(result[1], tuple(pt), 1, color)

                for i in range(len(result[0])):
                    #Find the points that are inside the bounding box of the object
                    in_BB_ind= points_in_BB(result[0], i, img_pts)
                    in_BB_XYZ = point_cloud_XYZ[in_BB_ind]

                    d_md = distance2object(in_BB_XYZ, method="mean_dist")
                    d_mz = distance2object(in_BB_XYZ, method="mean_z")
                    d_med = distance2object(in_BB_XYZ, method="median_dist")
                    d_mez = distance2object(in_BB_XYZ, method="median_z")

                    print("d_md : %.4f  , d_mz : %.4f, d_med : %.4f, d_mez : %.4f " % (d_md, d_mz, d_med, d_mez))

                    # Visualize pointcloud on the image with objects
                    if self.visualize and not self.visualize_all_points:
                        for idx, pt in enumerate(img_pts[in_BB_ind]):
                            dist = np.linalg.norm(in_BB_XYZ[idx])
                            color = depth_color(dist)
                            cv2.circle(result[1], tuple(pt), 1, color)

                self.out_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(result[1], 'bgr8'))

        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic , CameraInfo, self.image_info_callback) 
        self.synchronizer.registerCallback(callback)
        rospy.spin()

if __name__ == '__main__':

    node = Node()
    print("Detection started")
    node.run()
    
