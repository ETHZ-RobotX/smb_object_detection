#!/usr/bin/env python
from posixpath import join
import rospy
import ros_numpy
import numpy as np
import message_filters as mf
from numpy.lib.recfunctions import unstructured_to_structured

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from object_detection.msg import ObjectDetection, ObjectDetectionArray

from object_detection.objectdetector import ObjectDetector
from object_detection.pointprojector import PointProjector
from object_detection.objectlocalizer import ObjectLocalizer

from os.path import join

import warnings
warnings.filterwarnings("ignore")

def pointcloud2_to_xyzi(pointcloud2):
    pc_list = pc2.read_points_list(pointcloud2, skip_nans=True)
    xyzi = np.zeros((len(pc_list),4))
    for ind, p in enumerate(pc_list):
        xyzi[ind,0] = p[0]
        xyzi[ind,1] = p[1]
        xyzi[ind,2] = p[2]
        xyzi[ind,3] = p[3]
    return xyzi


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
        self.object_detection_pub_topic     = rospy.get_param('~object_detection_topic', '/objects')
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
        
        # ---------- Config Directory ----------
        self.config_dir                     = rospy.get_param('~config_dir', None)

        # ---------- Point Projector Related ---------- 
        self.project_cfg                  = rospy.get_param('~project_config', None)

        # ---------- 2D Object Detection Related ----------
        self.objectdetector_cfg             = rospy.get_param('~detector_config', None) 
        self.multiple_instance              = rospy.get_param('~multiple_instance', False)

        # ---------- 3D Object Localizer Related ----------
        self.objectlocalizer_cfg            = rospy.get_param('~localizer_config', None)
        self.objectlocalizer_save_data      = rospy.get_param('~objectlocalizer_save_data', False)
        self.objectlocalizer_learning_type  = rospy.get_param('~objectlocalizer_learning_type', None)

        # ---------- Objects of Actions ----------
        self.imagereader                    = CvBridge()
        self.pointprojector                 = PointProjector( join(self.config_dir, self.project_cfg))
        self.objectdetector                 = ObjectDetector( join(self.config_dir, self.objectdetector_cfg))           
        self.objectlocalizer                = ObjectLocalizer( join(self.config_dir, self.objectlocalizer_cfg) 
                                                             , self.config_dir, self.objectlocalizer_save_data
                                                             , self.objectlocalizer_learning_type)
        
        rospy.loginfo("Detector is set")
    

    def image_info_callback(self, camera_info):
        self.optical_frame_id  = camera_info.header.frame_id
        h                      = camera_info.height
        w                      = camera_info.width
        K                      = np.array(camera_info.K, dtype=np.float64).reshape(3,3)

        self.pointprojector.set_cameraparams(K, [w,h])

        if self.pointprojector.K is not None:
            self.camera_info_sub.unregister()

    def run(self):

        def callback(image_msg, lidar_msg):
            # If Image and Lidar messages are not empty 
            if image_msg.height > 0 and lidar_msg.width > 0:
                
                # Read lidar message
                # point_cloud_XYZ = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar_msg)
                # point_cloud_XYZ2 = pc2.read_points_list(lidar_msg)

                point_cloud_XYZ = pointcloud2_to_xyzi(lidar_msg)

                # Ground filter 
                # Upward direction is Z which 3rd column in the matrix
                # It is positive because it increases upwards
                point_cloud_XYZ = self.objectlocalizer.filter_ground(point_cloud_XYZ, 2)

                # translate and project PointCloud onto the Image  
                point_cloud_XYZ[:,:3] = self.pointprojector.translatePoints(point_cloud_XYZ[:,:3])
                pointcloud_on_image , in_FoV_indices = self.pointprojector.projectPointsOnImage(point_cloud_XYZ[:,:3])

                # transform the image msg to numpy array
                cv_image = self.imagereader.imgmsg_to_cv2(image_msg, "bgr8")

                # Detect objects in image
                object_detection_result, object_detection_image = self.objectdetector.detect(cv_image, multiple_instance = self.multiple_instance)
                
                # Localize every detected object
                object_poses_list, on_object_list = self.objectlocalizer.localize(object_detection_result, \
                                                                                  pointcloud_on_image, \
                                                                                  point_cloud_XYZ[in_FoV_indices], \
                                                                                  cv_image  )

                # Convert arrays to correct format
                pointcloud_on_image = np.c_[ pointcloud_on_image, np.zeros(pointcloud_on_image.shape[0]) ]
                pointcloud_on_image = unstructured_to_structured( pointcloud_on_image, dtype = np.dtype( [('x', np.float32), ('y', np.float32), ('z', np.float32)] ))
                pointcloud_in_FoV   = unstructured_to_structured( point_cloud_XYZ[in_FoV_indices], dtype = np.dtype( [('x', np.float32), ('y', np.float32), ('z', np.float32), ('i', np.float32)] ))
                
                # Create object detection message
                object_detection_array = ObjectDetectionArray()
                object_detection_array.header.stamp            = image_msg.header.stamp
                object_detection_array.header.frame_id         = self.optical_frame_id
                object_detection_array.header.seq              = self.seq
                object_detection_array.detections_image        = self.imagereader.cv2_to_imgmsg(object_detection_image, 'bgr8')
                object_detection_array.pointcloud_in_frame_2D  = ros_numpy.point_cloud2.array_to_pointcloud2(pointcloud_on_image) 
                object_detection_array.pointcloud_in_frame_3D  = ros_numpy.point_cloud2.array_to_pointcloud2(pointcloud_in_FoV) 
                
                self.seq = self.seq + 1 if self.seq < UINT32-1 else 0

                # For every detected image object
                for i in range(len(object_detection_result)):
                    object_detection = ObjectDetection()

                    object_detection.class_id = object_detection_result["name"][i]
                    object_detection.id       = 42 #TODO: Implement id 

                    object_detection.bounding_box_min_x = int(object_detection_result['xmin'][i])
                    object_detection.bounding_box_min_y = int(object_detection_result['ymin'][i])
                    object_detection.bounding_box_max_x = int(object_detection_result['xmax'][i])
                    object_detection.bounding_box_max_y = int(object_detection_result['ymax'][i])

                    object_detection.on_object_point_indices = list(on_object_list[i]) # if len(np.atleast_1d(on_object_list[i]))>1 else on_object_list[i]

                    object_detection.pose.x = object_poses_list[i][0]
                    object_detection.pose.y = object_poses_list[i][1]
                    object_detection.pose.z = object_poses_list[i][2]
                    
                    object_detection_array.detections.append(object_detection)
            
            # Publish the message
            self.object_detection_pub.publish(object_detection_array)

        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic , CameraInfo, self.image_info_callback)
        self.synchronizer.registerCallback(callback)

        rospy.spin()

if __name__ == '__main__':

    node = Node()
    rospy.loginfo("Detection has started")
    node.run()

    
