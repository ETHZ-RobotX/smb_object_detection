#!/usr/bin/env python
import rospy
import ros_numpy
import numpy as np
import message_filters as mf

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from object_detection.msg import ObjectDetection, ObjectDetectionArray

from object_detection.objectdetector import ObjectDetector
from object_detection.pointprojector import PointProjector
from object_detection.objectlocalizer import ObjectLocalizer
from object_detection.utils import *

import warnings
warnings.filterwarnings("ignore")


UINT32 = 2**32 - 1

# TODO: Add warnings such as "No self.optical_frame"

class Node:
    def __init__(self):
        # Initilized the node 
        rospy.init_node("objectify", anonymous=True)

        # ---------- Node Related Params Starts ---------- 
        # -> Subscribed Topics
        self.camera_topic                   = rospy.get_param('~camera_topic', '/versavis/cam0/undistorted')
        self.camera_info_topic              = rospy.get_param('~camera_info_topic', '/versavis/cam0/camera_info')
        self.lidar_topic                    = rospy.get_param('~lidar_topic', '/rslidar_points')                      

        # -> Published Topics
        self.object_detection_pub_topic     = rospy.get_param('~object_detection_topic', 'objects')
        self.object_detection_pub           = rospy.Publisher(self.object_detection_pub_topic , ObjectDetectionArray, queue_size=5)
        self.seq                            = 0
        
        # -> Topic Synchronization
        self.camera_lidar_sync_queue_size   = rospy.get_param('~camera_lidar_sync_queue_size', 10)
        self.camera_lidar_sync_slop         = rospy.get_param('~camera_lidar_sync_slop', 0.05)
        
        self.camera_sub                     = mf.Subscriber(self.camera_topic, Image)
        self.lidar_sub                      = mf.Subscriber(self.lidar_topic, PointCloud2)

        self.synchronizer                   = mf.ApproximateTimeSynchronizer([ self.camera_sub, self.lidar_sub], 
                                                                               self.camera_lidar_sync_queue_size,  
                                                                               self.camera_lidar_sync_slop)

        # ---------- Point Projector Related ---------- 
        self.reproject_cfg                  = rospy.get_param('~reproject_config', None)

        # ---------- 2D Object Detection Related ----------
        self.objectdetector_cfg             = rospy.get_param('~detector_config', None) 
        self.multiple_instance              = rospy.get_param('~multiple_instance', False)

        # ---------- 3D Object Localizer Related ----------
        self.objectlocalizer_cfg            = rospy.get_param('~localizer_config', None) 

        # ---------- Objects of Actions ----------
        self.imagereader                    = CvBridge()
        self.pointprojector                 = PointProjector(self.objectdetector_cfg)
        self.objectdetector                 = ObjectDetector(self.objectdetector_cfg)           
        self.objectlocalizer                = ObjectLocalizer(self.objectlocalizer_cfg)
        
        print("Object Detector is set")
    

    def image_info_callback(self, camera_info):
        self.optical_frame_id  = camera_info.header.frame_id
        h                   = camera_info.height
        w                   = camera_info.width
        K                   = np.array(camera_info.K, dtype=np.float64).reshape(3,3)

        self.pointprojector.set_cameraparams(K, [w,h])

        if self.pointprojector.K is not None:
            self.camera_info_sub.unregister()

    def run(self):

        def callback(image_msg, lidar_msg):
            # If Image and Lidar messages are not empty 
            if image_msg.height > 0 and lidar_msg.width > 0:
                
                # Read lidar message
                point_cloud_XYZ = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar_msg)

                # translate and project PointCloud onto the Image  
                point_cloud_XYZ = self.pointprojector.translatePoints(point_cloud_XYZ)
                pointcloud_on_image , pointcloud_in_FoV = self.pointprojector.projectPointsOnImage(point_cloud_XYZ)

                # transform the image msg to numpy array
                cv_image = self.imagereader.imgmsg_to_cv2(image_msg, "bgr8")

                # Detect objects in image
                object_detection_result, object_detection_image = self.objectdetector.detect(cv_image, multiple_instance = self.multiple_instance)
                
                # pick the one with highest confidance for every class.
                object_poses, on_object_list = self.objectlocalizer.localize(object_detection_result, \
                                                                             pointcloud_on_image, \
                                                                             pointcloud_in_FoV, \
                                                                             cv_image  )

                # Create object detection message
                object_detection_array = ObjectDetectionArray()
                object_detection_array.header.stamp         = image_msg.header.stamp
                object_detection_array.header.frame_id      = self.optical_frame_id
                object_detection_array.header.seq           = self.seq
                object_detection_array.detections_image     = self.imagereader.cv2_to_imgmsg(object_detection_image, 'bgr8')
                object_detection_array.pointcloud_in_BB_2D  = ros_numpy.point_cloud2.array_to_pointcloud2(pointcloud_on_image) # TODO: Implement 2D TEST
                object_detection_array.pointcloud_in_BB_3D  = ros_numpy.point_cloud2.array_to_pointcloud2(pointcloud_in_FoV) # TODO: Implement 3D TEST
                
                self.seq = self.seq + 1 if self.seq < UINT32-1 else 0

                # For every detected image object
                for i in range(len(self.object_detection_result)):
                    object_detection = ObjectDetection()

                    object_detection.class_id = object_detection_result["name"][i]
                    object_detection.id       = 42 #TODO: Implement id 

                    object_detection.bounding_box_min_x = object_detection_result['xmin'][i]
                    object_detection.bounding_box_min_y = object_detection_result['ymin'][i]
                    object_detection.bounding_box_max_x = object_detection_result['xmax'][i]
                    object_detection.bounding_box_max_y = object_detection_result['ymax'][i]

                    object_detection.on_object_point_indices = np.array(on_object_list[i], dtype=np.int32)
                    
                    object_detection.pose.pose.position.x = object_poses[i][0]
                    object_detection.pose.pose.position.y = object_poses[i][1]
                    object_detection.pose.pose.position.z = object_poses[i][2]
                    
                    # TODO: Add covariance maybe

                    object_detection_array.detections.append(object_detection)

            self.object_detection_pub.publish(object_detection_array)

        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic , CameraInfo, self.image_info_callback)
        self.synchronizer.registerCallback(callback)

        rospy.spin()

if __name__ == '__main__':

    node = Node()
    print("Detection started")
    node.run()
    node.dumpStats()

    
