#!/usr/bin/env python
import rospy
import ros_numpy
import numpy as np
from os.path import join
from numpy.lib.recfunctions import unstructured_to_structured

import message_filters as mf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, CameraInfo

from object_detection_msgs.msg import ObjectDetection, ObjectDetectionArray

from object_detection.objectdetector    import ObjectDetector
from object_detection.pointprojector    import PointProjector
from object_detection.objectlocalizer   import ObjectLocalizer
from object_detection.utils             import pointcloud2_to_xyzi, check_validity_image_info, filter_ground

import warnings
import time
warnings.filterwarnings("ignore")

UINT32      = 2**32 - 1 # might be unnecessary

class Node:
    def __init__(self):
        
        rospy.loginfo("[ObjectDetection Node] Object Detector initilization starts ...")

        # Initilized the node 
        rospy.init_node("objectify", anonymous=True)
        self.verbose                        = rospy.get_param('~verbose', True)

        # Flags to start object detection
        self.image_info_recieved            = False

        # ---------- Node Related Params ---------- 
        # -> Subscribed Topics
        self.camera_topic                   = rospy.get_param('~camera_topic', '/versavis/cam0/undistorted')
        self.camera_info_topic              = rospy.get_param('~camera_info_topic', '/versavis/cam0/camera_info')
        self.lidar_topic                    = rospy.get_param('~lidar_topic', '/rslidar_points')                      

        # -> Published Topics
        self.object_detection_pub_topic     = rospy.get_param('~object_detection_topic', '/objects')
        self.object_detection_pub           = rospy.Publisher(self.object_detection_pub_topic , ObjectDetectionArray, queue_size=1)
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
        self.objectdetector_cfg = {
            "architecture"      :  rospy.get_param('~architecture', 'yolo'),
            "model"             :  rospy.get_param('~model', 'yolov5n6').lower(),
            "device"            :  rospy.get_param('~device', 'cpu'),
            "confident"         :  rospy.get_param('~confident', '0.4'),
            "iou"               :  rospy.get_param('~iou', '0.1'),
            "checkpoint"        :  rospy.get_param('~checkpoint', None),
            "classes"           :  rospy.get_param('~classes', None),
            "multiple_instance" :  rospy.get_param('~multiple_instance', False)
        }

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
        
        # MAK tests
        # self.detection_counter = 0
        # self.accumulated_detection_time = 0
        
        rospy.loginfo("[ObjectDetection Node] Object Detector initilization done.")
        rospy.loginfo("[ObjectDetection Node] Waiting for image info ...")
        camera_info = rospy.wait_for_message(self.camera_info_topic , CameraInfo, timeout=10)
        self.image_info_callback(camera_info)

    # MAK note: I am actually fairly confused about what this function does.
    # It seems to be something used only for initialisation, thus I really do not think that having a dedicated subscriber makes much sense.
    def image_info_callback(self, camera_info):
        self.optical_frame_id  = "blackfly_right_optical_link" #camera_info.header.frame_id # change this back when included in camera info
        h                      = camera_info.height
        w                      = camera_info.width
        K                      = np.array(camera_info.K, dtype=np.float64).reshape(3,3)
        self.seq += 1

        if check_validity_image_info(K, w, h):
            self.pointprojector.set_intrinsic_params(K, [w,h])
            self.objectlocalizer.set_intrinsic_camera_param(K)
            rospy.loginfo("[ObjectDetection Node] Image info is set! Detection is starting in 1 sec!")
            rospy.sleep(1) # again, sleeping like 
            self.seq = 0
            self.image_info_recieved = True
            # self.camera_info_sub.unregister()
        else:
            # if self.seq > 20:
            #     rospy.logerr("[ObjectDetection Node] Image info could not be set after 20th try! Please check image info!")
            #     rospy.signal_shutdown("Image info missing!")
            # rospy.loginfo("[ObjectDetection Node] Image info is not set! Trying again after 1 sec!")
            # rospy.sleep(1) # this seems not smart, sleeping inside a callback.
            # the problem seems to be that two nodes publish to the camera_info topic.
            rospy.logerr(" ------------------ camera_info not valid ------------------------")

    def run(self):

        def callback(image_msg, lidar_msg):
            # If Image and Lidar messages are not empty
            if not image_msg.height > 0:
                rospy.logfatal("[ObjectDetection Node] Image message is empty. Object detecion is on hold.")
            if not lidar_msg.width > 0:
                rospy.logfatal("[ObjectDetection Node] Lidar message is empty. Object detecion is on hold.")

            if self.image_info_recieved and image_msg.height > 0 and lidar_msg.width > 0:
                
                # Read lidar message
                point_cloud_XYZ = pointcloud2_to_xyzi(lidar_msg)

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
                # detection_start = time.time()
                object_detection_result, object_detection_image = self.objectdetector.detect(cv_image)
                # detection_end = time.time()
                # self.detection_counter += 1
                # self.accumulated_detection_time += detection_end-detection_start
                # rospy.loginfo(f"average detection took: {self.accumulated_detection_time/self.detection_counter}")
                # Localize every detected object
                object_list = self.objectlocalizer.localize(object_detection_result, \
                                                            pointcloud_on_image, \
                                                            point_cloud_XYZ[in_FoV_indices], \
                                                            cv_image  )
 
                # Create object detection message
                object_detection_array = ObjectDetectionArray()
                object_detection_array.header.stamp            = image_msg.header.stamp
                object_detection_array.header.frame_id         = self.optical_frame_id
                object_detection_array.header.seq              = self.seq
                self.seq = self.seq + 1 if self.seq < UINT32-1 else 0

                # For every detected image object, fill the message object. 
                for i in range(len(object_detection_result)):
                    object_detection = ObjectDetection()

                    object_detection.class_id = object_detection_result["name"][i]
                    object_detection.id       = object_list[i].id

                    object_detection.pose_estimation_type = object_list[i].estimation_type 

                    object_detection.pose.x = object_list[i].pos[0]
                    object_detection.pose.y = object_list[i].pos[1]
                    object_detection.pose.z = object_list[i].pos[2]

                    if self.verbose:
                        object_detection.bounding_box_min_x = int(object_detection_result['xmin'][i])
                        object_detection.bounding_box_min_y = int(object_detection_result['ymin'][i])
                        object_detection.bounding_box_max_x = int(object_detection_result['xmax'][i])
                        object_detection.bounding_box_max_y = int(object_detection_result['ymax'][i])

                        object_detection.on_object_point_indices = list(object_list[i].pt_indices) 
                        
                    object_detection_array.detections.append(object_detection)

                # Only if verbose true
                if self.verbose:
                    # Convert arrays to correct format
                    pointcloud_on_image = np.c_[ pointcloud_on_image, np.zeros(pointcloud_on_image.shape[0]) ]
                    pointcloud_on_image = unstructured_to_structured( pointcloud_on_image, dtype = np.dtype( [('x', np.float32), ('y', np.float32), ('z', np.float32)] ))
                    pointcloud_in_FoV   = unstructured_to_structured( point_cloud_XYZ[in_FoV_indices], dtype = np.dtype( [('x', np.float32), ('y', np.float32), ('z', np.float32), ('i', np.float32)] ))
                    # Create verbose message fields
                    object_detection_array.detections_image        = self.imagereader.cv2_to_imgmsg(object_detection_image, 'bgr8')
                    object_detection_array.pointcloud_in_frame_2D  = ros_numpy.point_cloud2.array_to_pointcloud2(pointcloud_on_image) 
                    object_detection_array.pointcloud_in_frame_3D  = ros_numpy.point_cloud2.array_to_pointcloud2(pointcloud_in_FoV) 

                # Publish the message
                self.object_detection_pub.publish(object_detection_array)

        # self.camera_info_sub = rospy.Subscriber(self.camera_info_topic , CameraInfo, self.image_info_callback)
        self.synchronizer.registerCallback(callback)

        rospy.spin()

if __name__ == '__main__':
    node = Node()
    rospy.loginfo("[ObjectDetection Node] Detection has started")
    node.run()
    
# change the queue sizes
