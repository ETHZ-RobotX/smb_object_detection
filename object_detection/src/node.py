#!/usr/bin/env python
import rospy
import rospkg
import ros_numpy
import numpy as np

from os.path import join
from numpy.lib.recfunctions import unstructured_to_structured

import message_filters as mf
import sklearn
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, CameraInfo

from object_detection_msgs.msg import   PointCloudArray,\
                                        ObjectDetectionInfo, ObjectDetectionInfoArray

from std_msgs.msg import Header

from geometry_msgs.msg import PoseArray, Pose, Quaternion

from object_detection.objectdetector    import ObjectDetector
from object_detection.pointprojector    import PointProjector
from object_detection.objectlocalizer   import ObjectLocalizer
from object_detection.utils             import *

import warnings
import time
warnings.filterwarnings("ignore")

UINT32      = 2**32 - 1 # might be unnecessary

class Node:
    def __init__(self):
        
        rospy.loginfo("[ObjectDetection Node] Object Detector initilization starts ...")
        self.rospack = rospkg.RosPack()

        # Initilized the node 
        rospy.init_node("objectify", anonymous=True)
        self.verbose                        = rospy.get_param('~verbose', True)
        self.project_object_points_to_img   = rospy.get_param('~project_object_points_to_image', True)
        self.project_all_points_to_img      = rospy.get_param('~project_all_points_to_image', False)

        # Flags to start object detection
        self.image_info_recieved            = False

        # ---------- Node Related Params ---------- 
        # -> Subscribed Topics
        self.camera_topic                   = rospy.get_param('~camera_topic', '/versavis/cam0/undistorted')
        self.camera_info_topic              = rospy.get_param('~camera_info_topic', '/versavis/cam0/camera_info')
        self.lidar_topic                    = rospy.get_param('~lidar_topic', '/rslidar_points')                      

        # -> Published Topics
        # self.object_detection_pub_topic     = rospy.get_param('~object_detection_topic', '/objects')
        self.object_detection_pub_pose_topic = rospy.get_param('~object_detection_pose_topic', '~object_poses')
        self.object_detection_pub_img_topic = rospy.get_param('~object_detection_output_image_topic', '~detections_in_image')
        self.object_detection_pub_pts_topic = rospy.get_param('~object_detection_point_clouds_topic', '~detection_point_clouds')
        self.object_detection_info_topic    = rospy.get_param('~object_detection_info_topic', '~detection_info')

        # self.object_detection_pub           = rospy.Publisher(self.object_detection_pub_topic , ObjectDetectionArray, queue_size=1)
        self.object_pose_pub                = rospy.Publisher(self.object_detection_pub_pose_topic , PoseArray, queue_size=1)
        self.object_detection_img_pub       = rospy.Publisher(self.object_detection_pub_img_topic , Image, queue_size=1)
        self.object_point_clouds_pub        = rospy.Publisher(self.object_detection_pub_pts_topic , PointCloudArray, queue_size=1)
        self.detection_info_pub             = rospy.Publisher(self.object_detection_info_topic , ObjectDetectionInfoArray, queue_size=1)
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
        self.config_dir                     = join(self.rospack.get_path('object_detection'),'cfg')

        # ---------- Point Projector Related ---------- 
        self.project_cfg                  = rospy.get_param('~project_config', None)

        # ---------- 2D Object Detection Related ----------
        self.objectdetector_cfg = {
            "architecture"      :  rospy.get_param('~architecture', 'yolo'),
            "model"             :  rospy.get_param('~model', 'yolov5n6').lower(),
            "model_path"        :  rospy.get_param('~model_path', ''),
            "device"            :  rospy.get_param('~device', 'cpu'),
            "confident"         :  rospy.get_param('~confident', '0.4'),
            "iou"               :  rospy.get_param('~iou', '0.1'),
            "checkpoint"        :  rospy.get_param('~checkpoint', None),
            "classes"           :  rospy.get_param('~classes', None),
            "multiple_instance" :  rospy.get_param('~multiple_instance', False)
        }

        print("cfg device: ", self.objectdetector_cfg["device"])

        # ---------- 3D Object Localizer Related ----------
        self.objectlocalizer_cfg = {
            "model_method"                  :  rospy.get_param('~model_method', 'hdbscan'),
            "ground_percentage"             :  rospy.get_param('~ground_percentage', 25),
            "bb_contract_percentage"        :  rospy.get_param('~bb_contract_percentage', 10),
            "distance_estimator_type"       :  rospy.get_param('~distance_estimator_type', 'none'),
            "distance_estimator_save_data"  :  rospy.get_param('~distance_estimator_save_data', 'False'),
            "object_specific_file"          :  rospy.get_param('~object_specific_file', 'object_specific.yaml'),
            "min_cluster_size"              :  rospy.get_param('~min_cluster_size', 5),
            "cluster_selection_epsilon"     :  rospy.get_param('~cluster_selection_epsilon', 0.08),
        }

        # ---------- Objects of Actions ----------
        self.imagereader                    = CvBridge()
        self.pointprojector                 = PointProjector( join(self.config_dir, self.project_cfg))
        self.objectdetector                 = ObjectDetector( self.objectdetector_cfg )           
        self.objectlocalizer                = ObjectLocalizer( self.objectlocalizer_cfg, self.config_dir )
        
        
        rospy.loginfo("[ObjectDetection Node] Object Detector initilization done.")
        rospy.loginfo("[ObjectDetection Node] Waiting for image info ...")
        rospy.loginfo(f"[ObjectDetection Node] If this takes longer than a few seconds, make sure {self.camera_info_topic} is published.")
        camera_info = rospy.wait_for_message(self.camera_info_topic , CameraInfo)
        self.image_info_callback(camera_info)

    def image_info_callback(self, camera_info):
        self.optical_frame_id  = camera_info.header.frame_id # "blackfly_right_optical_link" 
        h                      = camera_info.height
        w                      = camera_info.width
        K                      = np.array(camera_info.K, dtype=np.float64).reshape(3,3)
        self.seq += 1

        if check_validity_image_info(K, w, h):
            self.pointprojector.set_intrinsic_params(K, [w,h])
            self.objectlocalizer.set_intrinsic_camera_param(K)
            rospy.loginfo("[ObjectDetection Node] Image info is set! Detection is starting in 1 sec!")
            rospy.sleep(1)
            self.seq = 0
            self.image_info_recieved = True
        else:
            rospy.logerr(" ------------------ camera_info not valid ------------------------")

    def run(self):

        def callback(image_msg, lidar_msg):
            callback_start = time.time()
            # If Image and Lidar messages are not empty
            if not image_msg.height > 0:
                rospy.logfatal("[ObjectDetection Node] Image message is empty. Object detecion is on hold.")
            if not lidar_msg.width > 0:
                rospy.logfatal("[ObjectDetection Node] Lidar message is empty. Object detecion is on hold.")

            if self.image_info_recieved and image_msg.height > 0 and lidar_msg.width > 0:
                
                # Read lidar message
                point_cloud_XYZ = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar_msg)

                # Ground filter 
                # Upward direction is Z which 3rd column in the matrix
                # It is positive because it increases upwards
                point_cloud_XYZ = filter_ground(point_cloud_XYZ, self.objectlocalizer_cfg["ground_percentage"])

                # translate and project PointCloud onto the Image  
                point_cloud_XYZ[:,:3] = self.pointprojector.transform_points(point_cloud_XYZ[:,:3])
                pointcloud_on_image , in_FoV_indices = self.pointprojector.project_points_on_image(point_cloud_XYZ[:,:3])


                # transform the image msg to numpy array
                cv_image = self.imagereader.imgmsg_to_cv2(image_msg, "bgr8")

                # Detect objects in image
                object_detection_result, object_detection_image = self.objectdetector.detect(cv_image)

                # Localize every detected object
                object_list = self.objectlocalizer.localize(object_detection_result, \
                                                            pointcloud_on_image, \
                                                            point_cloud_XYZ[in_FoV_indices], \
                                                            cv_image  )
                
                header = Header()
                header.stamp = image_msg.header.stamp
                header.frame_id = self.optical_frame_id
                header.seq = self.seq
                self.seq = self.seq + 1 if self.seq < UINT32-1 else 0

                object_pose_array = PoseArray()
                object_pose_array.header = header
                object_information_array = ObjectDetectionInfoArray()
                object_information_array.header = header
                point_cloud_array = PointCloudArray()
                point_cloud_array.header = header
                 
                pointcloud_in_FoV = point_cloud_XYZ[in_FoV_indices]
                                                                                
                # For every detected image object, fill the message object. 
                for i in range(len(object_detection_result)):
                    object_pose = Pose()
                    object_information = ObjectDetectionInfo()
                    object_point_cloud = PointCloud2()

                    object_information.class_id = object_detection_result["name"][i]
                    object_pose.position.x = object_list[i].pos[0]
                    object_pose.position.y = object_list[i].pos[1]
                    object_pose.position.z = object_list[i].pos[2]
                    object_pose.orientation = Quaternion(0,0,0,1)

                    object_information.position.x = object_list[i].pos[0]
                    object_information.position.y = object_list[i].pos[1]
                    object_information.position.z = object_list[i].pos[2]
                    object_information.id       = object_list[i].id
                    object_information.pose_estimation_type = object_list[i].estimation_type
                    object_information.confidence = object_detection_result["confidence"][i]

                    object_information.bounding_box_min_x = int(object_detection_result['xmin'][i])
                    object_information.bounding_box_min_y = int(object_detection_result['ymin'][i])
                    object_information.bounding_box_max_x = int(object_detection_result['xmax'][i])
                    object_information.bounding_box_max_y = int(object_detection_result['ymax'][i])

                    object_point_cloud = pointcloud_in_FoV[object_list[i].pt_indices,:]
                    object_point_cloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(unstructured_to_structured(object_point_cloud, 
                                                                dtype = np.dtype( [('x', np.float32), ('y', np.float32), ('z', np.float32)] )))

                    if (not self.project_all_points_to_img) and self.project_object_points_to_img:
                        for idx, pt in enumerate(pointcloud_on_image[object_list[i].pt_indices,:]):
                            dist = object_point_cloud[idx,2]
                            color = depth_color(dist, min_d=0.5, max_d=20)
                            try:
                                cv2.circle(object_detection_image, pt[:2].astype(np.int32), 2, color, -1 )
                            except:
                                print("Cannot Circle \n")    

                    object_pose_array.poses.append(object_pose)
                    object_information_array.info.append(object_information)
                    point_cloud_array.point_clouds.append(object_point_cloud_msg)

                if self.project_all_points_to_img:
                    for idx, pt in enumerate(pointcloud_on_image): 
                        dist = pointcloud_in_FoV[idx,2]
                        color = depth_color(dist, min_d=0.5, max_d=30)
                        try:
                            cv2.circle(object_detection_image, pt[:2].astype(np.int32), 3, color, -1)
                        except:
                            print("Cannot Circle \n")    

                # Publish the message
                self.object_pose_pub.publish(object_pose_array)
                self.detection_info_pub.publish(object_information_array)
                self.object_point_clouds_pub.publish(point_cloud_array)
                self.object_detection_img_pub.publish(self.imagereader.cv2_to_imgmsg(object_detection_image, 'bgr8'))
                

        self.synchronizer.registerCallback(callback)

        rospy.spin()

if __name__ == '__main__':
    node = Node()
    rospy.loginfo("[ObjectDetection Node] Detection has started")
    node.run()

