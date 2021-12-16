#!/usr/bin/env python
import rospy
import ros_numpy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, TransformStamped
from apriltag_ros.msg import *
from cv_bridge import CvBridge
import cv2
import tf2_ros

from object_detection.objectdetector import ObjectDetector
from object_detection.reproject import ImageHandler
from object_detection.objectlocalizer import *

from datetime import datetime

from time import sleep, perf_counter

import warnings
warnings.filterwarnings("ignore")

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

def depth_color(val, min_d=0.2, max_d=20):
    """
    print Color(HSV's H value) corresponding to distance(m)
    close distance = red , far distance = blue
    """
    # np.clip(val, 0, max_d, out=val)

    hsv = (-1*((val - min_d) / (max_d - min_d)) * 255).astype(np.uint8)
    return hsv2rgb(hsv,1,1)

def marker_(ns, marker_id, pos, stamp ,color, type="all"):

    marker = Marker()
    marker.ns = str(ns)
    marker.header.frame_id = "map"
    marker.header.stamp = stamp
    marker.type = 2
    marker.action = 0
    marker.pose = Pose()

    marker.pose.position.x = pos[0]
    marker.pose.position.y = pos[1]
    marker.pose.position.z = pos[2]

    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0

    if type=="mean":
        marker.id = 10000

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 1.0
    else:
        
        marker.id = marker_id
        
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.3


    marker.frame_locked = False
        
    return marker

def transformstamped_(frame_id, child_id, time, pose, rot):
    t = TransformStamped()

    t.header.stamp = time
    t.header.frame_id = frame_id
    t.child_frame_id = child_id
    t.transform.translation.x = pose[0]
    t.transform.translation.y = pose[1]
    t.transform.translation.z = pose[2]
    t.transform.rotation.x = rot[0]
    t.transform.rotation.y = rot[1]
    t.transform.rotation.z = rot[2]
    t.transform.rotation.w = rot[3]

    return t

def pointcloud_filter_ground(xyz):
    min_z = min(xyz[:,2])

    non_ground_indices = np.argwhere(xyz[:,2] > min_z*0.85).flatten()
    return(xyz[non_ground_indices])

def param_array_parser(param_array):
    if param_array != 'None':
        classes = np.array([[int(x.strip(' ')) for x in ss.lstrip(' [,').split(', ')] for ss in param_array.rstrip(']').split(']')])
        return list(classes.flatten())
    else:
        return None

class Node:
    def __init__(self):

        rospy.init_node("objectify", anonymous=True)

        # Node related
        self.camera_topic                   = rospy.get_param('~camera_topic', '/versavis/cam0/undistorted')
        self.camera_info_topic              = rospy.get_param('~camera_info_topic', '/versavis/cam0/camera_info')
        self.lidar_topic                    = rospy.get_param('~lidar_topic', '/rslidar_points')

        self.camera_info_callback_sleep     = rospy.get_param('~camera_info_callback_sleep', 20)

        self.optical_frame                  = rospy.get_param('~optical_frame', 'blackfly_right_optical_link')
        self.map_frame                      = rospy.get_param('~map_frame', 'map')

        self.camera_sub                     = message_filters.Subscriber(self.camera_topic, Image)
        self.lidar_sub                      = message_filters.Subscriber(self.lidar_topic, PointCloud2)

        self.cv_bridge                      = CvBridge()

        self.synchronizer                   = message_filters.ApproximateTimeSynchronizer([self.camera_sub, self.lidar_sub], 30, 1, reset=True)

        self.validate                       = rospy.get_param('~validate', True)
        self.record_data                    = rospy.get_param('~record_data', True)

        self.tf_buffer                      = tf2_ros.Buffer(rospy.Duration(4.0)) #tf buffer length
        self.tf_listener                    = tf2_ros.TransformListener(self.tf_buffer)
        
        # Validate the location with AprilTag
        self.mean_tag_pos                   = None
        if self.validate :
            self.tag_topic                      = rospy.get_param('~tag_topic', '/tag_detections')
            self.on_object_tag                  = rospy.get_param('~on_object_tag', 0)
            self.tag_pos                        = np.empty((0,3))
            
        # Detection related 
        self.multiple_instance              = rospy.get_param('~multiple_instance', False)

        # Object Detector related
        self.objectdetector_cfg = {}
        self.objectdetector_cfg['architecture']   = rospy.get_param('~architecture', 'yolo')
        self.objectdetector_cfg['model']          = rospy.get_param('~model', 'yolov5n')
        self.objectdetector_cfg['checkpoint']     = rospy.get_param('~checkpoint', None)
        self.objectdetector_cfg['device']         = rospy.get_param('~device', 'cpu')
        self.objectdetector_cfg['confident']      = rospy.get_param('~confident', 0.5)
        self.objectdetector_cfg['iou']            = rospy.get_param('~iou', 0.45)
        self.objectdetector_cfg['classes']        = param_array_parser( rospy.get_param('~classes',  None) )          
       
        # Object Localizaer related 
        self.objectlocalizer_cfg = {}
        self.objectlocalizer_cfg['model_method']        = rospy.get_param('~model_method', 'hdbscan') # same class multiple instance
        self.objectlocalizer_cfg['localizer_config']    = rospy.get_param('~localizer_config', None) # same class multiple instance


        # Output related
        self.visualize                      = rospy.get_param('~visualize', True)
        self.visualize_all_points           = rospy.get_param('~visualize_all_points', False)
        
        if self.visualize:
            self.out_image_pub_topic            = rospy.get_param('~out_image_pub_topic', "/versavis/cam0/objects")
            self.out_image_pub                  = rospy.Publisher(self.out_image_pub_topic , Image, queue_size=5)
        
        self.TF_br                          = tf2_ros.StaticTransformBroadcaster()

        self.create_obj_marker              = rospy.get_param('~create_obj_marker', 'true')
        self.obj_marker_topic               = rospy.get_param('~obj_marker_topic', '/object')

        if self.create_obj_marker:
            self.marker_pub                     = rospy.Publisher(self.obj_marker_topic , MarkerArray, queue_size=10)
            self.obj_id                         = 0
            self.marker_color                   = {}

            if self.objectdetector_cfg['classes'] is not None:
                for c in self.objectdetector_cfg['classes']:
                    self.marker_color [c] = np.random.rand(3)


        # By-products
        self.object_detection_result        = None
        self.pointcloud_on_image            = None      # 2D pointcloud on image with pixel coordinages 
        self.pointcloud_in_image            = None      # 3D pointcloud in image frame
        self.cv_image                       = None

        # Camera Params
        self.imagehandler                   = ImageHandler()


        if self.record_data:
            self.pos_error          = []
            self.gt_pos             = np.array([14.9493683 , -2.11283953])
            self.statistic_txt      = rospy.get_param('~statistic_txt', 'asd') 



        # -------- TODO: REMOVE HARDCODED PARAMS ---------------------------

        ##  rosrun tf tf_echo "rslidar" "blackfly_right_optical_link" 1
        ## - Translation: [-0.045, -0.293, -0.241]
        ## - Rotation: in Quaternion [0.000, -0.707, 0.707, -0.000]
        ##             in RPY (radian) [-1.570, -0.000, -3.141]
        ##             in RPY (degree) [-89.954, -0.000, -179.954]

        R_camera_lidar = np.float64([ [-1.0000000, 0.0000000,  0.0000000],
                                        [0.0000000, 0.0000000, -1.0000000],
                                        [0.0000000, -1.0000000,  0.0000000] ])
        # IMU correction was: 7.44614 deg (0.99789, 0.0518968, -0.021901, 0.0323034)
        # [ x: 176.1699033, y: -2.3128941, z: -6.0314909 ]
        # Correction of Lidar position
        R = [ [ 0.9998379,  0.0180019,  0.0000000],
              [-0.0180019,  0.9998379,  0.0000000],
              [ 0.0000000,  0.0000000,  1.0000000]]

        R_camera_lidar = np.matmul(R_camera_lidar,R)       
        t_camera_lidar = np.float64([-0.045, -0.293, -0.241])

        # -------- TODO: REMOVE HARDCODED PARAMS ---------------------------

        self.imagehandler.set_transformationparams(R_camera_lidar,t_camera_lidar)
        
        self.objectlocalizer     = ObjectLocalizer(self.objectlocalizer_cfg)
        self.objectdetector      = ObjectDetector(self.objectdetector_cfg)
        print("Object Detector is set")
    
    def dumpStats(self):
       
        print("Mean tag pose: " + str(self.mean_tag_pos) + "\n")

        self.pos_error = np.array(self.pos_error)

        now  = datetime.now()
        now  = now.strftime("%H_%M_%S")

        with open("/home/oilter/Documents/Statistics/" + self.statistic_txt + ".txt", "w") as record:
            record.write("Statistic of Localization of Objects \n")
            record.write("Date and Time: " + now + "\n")
            record.write("\n")

            record.write("!!!!!! \n")
            record.write("Discarding errors more than 2 meters while calculating stats\n")
            record.write("!!!!!! \n")

            error_2     = self.pos_error < 2.0

            record.write("Pos Error \n")
            record.write("Mean: " + str(np.mean(self.pos_error[error_2])) + "\n")
            record.write("Median: " + str(np.median(self.pos_error[error_2])) + "\n")
            record.write("Std: " + str(np.std(self.pos_error[error_2])) + "\n")
            record.write("Max Error in 2 m: " + str(max(self.pos_error[error_2])) + "\n")
            record.write("Max Error in whole: " + str(max(self.pos_error)) + "\n")
            
            
            record.write("\n")
            record.write("Number of whole samples: " + str(len(self.pos_error)) + "\n")
            
            error_2     = self.pos_error < 2.0
            error_15     = self.pos_error < 1.5
            error_1     = self.pos_error < 1.0
            error_05    = self.pos_error < 0.5
            error_025   = self.pos_error < 0.25
            error_01   = self.pos_error < 0.1

            record.write("Number of samples that has error less than 2.0 m: " + str(len(self.pos_error[error_2])) + "\n")
            record.write("Number of samples that has error less than 1.5 m: " + str(len(self.pos_error[error_15])) + "\n")
            record.write("Number of samples that has error less than 1.0 m: " + str(len(self.pos_error[error_1])) + "\n")
            record.write("Number of samples that has error less than 0.5 m: " + str(len(self.pos_error[error_05])) + "\n")
            record.write("Number of samples that has error less than 0.25 m: " + str(len(self.pos_error[error_025])) + "\n")
            record.write("Number of samples that has error less than 0.1 m: " + str(len(self.pos_error[error_01])) + "\n")

            record.write("Mean tag pose: " + str(self.mean_tag_pos) + "\n")

            record.write("------------------------------------------------------------- \n")

        print("Statistic of Localization of Objects \n")
        print("Date and Time: " + now + "\n")
        print("\n")

        print("!!!!!! \n")
        print("Discarding errors more than 2 meters while calculating stats\n")
        print("!!!!!! \n")

        print("Pos Error \n")
        print("Mean: " + str(np.mean(self.pos_error[error_2])) + "\n")
        print("Median: " + str(np.median(self.pos_error[error_2])) + "\n")
        print("Std: " + str(np.std(self.pos_error[error_2])) + "\n")
        print("Max Error in 2 m: " + str(max(self.pos_error[error_2])) + "\n")
        print("Max Error in whole: " + str(max(self.pos_error)) + "\n")
                
        print("\n")
        print("Number of whole samples: " + str(len(self.pos_error)) + "\n")

        print("Number of samples that has error less than 2.0 m: " + str(len(self.pos_error[error_2])) + "\n")
        print("Number of samples that has error less than 1.5 m: " + str(len(self.pos_error[error_15])) + "\n")
        print("Number of samples that has error less than 1.0 m: " + str(len(self.pos_error[error_1])) + "\n")
        print("Number of samples that has error less than 0.5 m: " + str(len(self.pos_error[error_05])) + "\n")
        print("Number of samples that has error less than 0.25 m: " + str(len(self.pos_error[error_025])) + "\n")
        print("Number of samples that has error less than 0.1 m: " + str(len(self.pos_error[error_01])) + "\n")

        print("Mean tag pose: " + str(self.mean_tag_pos) + "\n")

        print("------------------------------------------------------------- \n")

    def image_info_callback(self, camera_info):
        h = camera_info.height
        w = camera_info.width
        K = np.array(camera_info.K, dtype=np.float64).reshape(3,3)

        self.imagehandler.set_cameraparams(K, [w,h])
        rospy.sleep(self.camera_info_callback_sleep)

    def tag_callback(self, data):
        for tag in data.detections:
            # print("Tag ",tag.id[0]," has been seen." )

            if tag.id[0] != self.on_object_tag:
                continue

            try:
                transform = self.tf_buffer.lookup_transform(self.map_frame,
                                            'tag_' + str(tag.id[0]), #source frame
                                            rospy.Time(0),
                                            rospy.Duration(1)) #get the tf at first available time) #wait for 5 second
                xyz = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z ])
                self.tag_pos = np.vstack([self.tag_pos, xyz])
                self.mean_tag_pos = np.mean(self.tag_pos, axis=0)
                self.mean_tag_pos = xyz

            except:
                print("Could not transform the tag pos")

    def run(self):

        def pointcloud_on_image(lidar_msg):
            # transform the pointcloud msg to numpy array and remove nans
            point_cloud_XYZ = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar_msg)

            # point_cloud_XYZ = pointcloud_filter_ground(point_cloud_XYZ)

            # translate and project PointCloud onto the Image
            # 3D Lidar points that are inside the image frame  
            point_cloud_XYZ = self.imagehandler.translatePoints(point_cloud_XYZ)
            self.pointcloud_on_image , self.pointcloud_in_image = self.imagehandler.projectPointsOnImage(point_cloud_XYZ)
        
        def object_detection(image_msg):
            # transform the image msg to numpy array
            self.cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # Detect objects in image
            self.object_detection_result = self.objectdetector.detect(self.cv_image, return_image=self.visualize)
            # If multiple instance for classes is not allowed,
            # pick the one with highest confidance for every class.
            if not self.multiple_instance:
                detected_objects = []
                row_to_delete    = []
                for i in range(len(self.object_detection_result[0])):
                    if self.object_detection_result[0]['class'][i] in detected_objects:
                        row_to_delete.append(i)
                    else:
                        detected_objects.append(self.object_detection_result[0]['class'][i])
                
                self.object_detection_result[0] = self.object_detection_result[0].drop(row_to_delete, axis=0)
                self.object_detection_result[0].reset_index(inplace=True)

        def callback(image_msg, lidar_msg):
            # If Image and Lidar messages are not empty 
            if image_msg.height != 0 and lidar_msg.width > 0:
                
                # start_time = perf_counter()
                pointcloud_on_image(lidar_msg)
                # end_time = perf_counter()
                # self.time_img_hand.append(end_time - start_time)

                # start_time = perf_counter()
                object_detection(image_msg)                    
                # end_time = perf_counter()
                # self.time_obj_det.append(end_time - start_time)

                # print(self.object_detection_result[0])

                # start_time = perf_counter()
                object_poses, on_object_list = self.objectlocalizer.localize(self.object_detection_result[0], \
                                                                             self.pointcloud_on_image, \
                                                                             self.pointcloud_in_image, \
                                                                             self.cv_image  )
                # end_time = perf_counter()
                # self.time_obj_pos.append(end_time - start_time)

                # if(len(object_poses) == 0):
                #     return

                # Visualize all Lidar points that are inside the image frame
                if self.visualize and self.visualize_all_points:
                    for idx, pt in enumerate(self.pointcloud_on_image):
                        dist = np.linalg.norm(self.pointcloud_in_image[idx])
                        color = depth_color(dist)
                        cv2.circle(self.object_detection_result[1], tuple(pt), 1, color)


                # For every detected image object
                for i in range(len(self.object_detection_result[0])):

                    try:
                        xyz = object_poses[i]
                    except:
                        continue

                    if np.any(np.isnan(xyz)):
                        continue
                    
                    # start_time = perf_counter()
                    # transformstamped_("blackfly_right_optical_link", "object", xyz, [0,0,0,1])
                    self.TF_br.sendTransform(transformstamped_(self.optical_frame, "object", \
                                                                image_msg.header.stamp, xyz, [0,0,0,1]))
                    try:
                        transform = self.tf_buffer.lookup_transform_full(self.map_frame, image_msg.header.stamp, \
                                                                        "object", image_msg.header.stamp, self.map_frame, \
                                                                        rospy.Duration(2.0))
                    except:
                        print("cannot transform")
                        continue


                    # end_time = perf_counter()
                    # self.time_obj_map_pos.append(end_time - start_time)


                    obj_in_map = np.array([transform.transform.translation.x, \
                                           transform.transform.translation.y, \
                                           transform.transform.translation.z ])
                                      
                    if self.create_obj_marker:
                        markers = MarkerArray()
                        markers.markers.append(marker_("object", self.obj_id, obj_in_map, image_msg.header.stamp, [1.0,0,0]))
                        if self.mean_tag_pos is not None and self.validate:
                            markers.markers.append(marker_("tag", 1, self.mean_tag_pos, image_msg.header.stamp, [0,1.0,0]))
                        self.marker_pub.publish(markers)       

                    self.pos_error.append(np.linalg.norm( self.gt_pos - obj_in_map[:2] ))      
                    # self.pos_error.append(np.linalg.norm( self.mean_tag_pos - obj_in_map[:2] ))      

                    self.obj_id += 1
                    # Visualize pointcloud on the image with objects
                    if self.visualize and not self.visualize_all_points:
                        for idx, pt in enumerate( self.pointcloud_on_image[on_object_list[i],:]): 
                            dist = np.linalg.norm(self.pointcloud_in_image[on_object_list[i]])
                            color = depth_color(dist)
                            try:
                                cv2.circle(self.object_detection_result[1], pt, 1, color)
                            except:
                                print("Cannot circle")
                                print(pt)

                if self.visualize:
                    img_msg = self.cv_bridge.cv2_to_imgmsg(self.object_detection_result[1], 'bgr8')
                    img_msg.header.frame_id = self.optical_frame
                    self.out_image_pub.publish(img_msg)
                
                # end_time = perf_counter()
                # print(f'1) It took {end_time- start_time: 0.4f} second(s) to complete.')

        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic , CameraInfo, self.image_info_callback)
        self.synchronizer.registerCallback(callback)

        if self.validate:
            self.tag_sub         = rospy.Subscriber(self.tag_topic , AprilTagDetectionArray, self.tag_callback) 

        rospy.spin()

if __name__ == '__main__':

    node = Node()
    print("Detection started")
    node.run()
    node.dumpStats()

    
