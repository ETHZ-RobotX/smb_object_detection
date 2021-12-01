#!/usr/bin/env python
from ros_numpy import point_cloud2
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

def marker_(ns, marker_id, pos, color, type="all"):

    marker = Marker()
    marker.ns = str(ns)
    marker.header.frame_id = "map"
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

def transformstamped_(frame_id, child_id, pose, rot):
    t = TransformStamped()

    t.header.stamp = rospy.Time.now()
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
        self.out_image_pub_topic            = rospy.get_param('~out_image_pub_topic', "/versavis/cam0/objects")

        self.out_image_pub                  = rospy.Publisher(self.out_image_pub_topic , Image, queue_size=5)
        self.TF_br                          = tf2_ros.StaticTransformBroadcaster()

        self.obj_marker_topic               = rospy.get_param('~obj_marker_topic', '/object')
        self.marker_pub                     = rospy.Publisher(self.obj_marker_topic , MarkerArray, queue_size=10)

        # Verify the location with AprilTag
        self.tf_buffer                      = tf2_ros.Buffer(rospy.Duration(0.5)) #tf buffer length
        self.tf_listener                    = tf2_ros.TransformListener(self.tf_buffer)
        self.tag_topic                      = rospy.get_param('~tag_topic', '/tag_detections')
        self.tag_pos                        = np.empty((0,3))
        self.mean_tag_pos                   = None
        self.obj_id                         = 0

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
        classes                             = rospy.get_param('~classes',  None)
        if classes != 'None':
            classes = np.array([[int(x.strip(' ')) for x in ss.lstrip(' [,').split(', ')] for ss in classes.rstrip(']').split(']')])
            self.objectdetector_cfg['classes'] = list(classes.flatten())
        else:
            self.objectdetector_cfg['classes'] = None
        
        # Object Localizaer related 
        self.objectlocalizer_cfg = {}
        self.objectlocalizer_cfg['model_method']   = rospy.get_param('~model_method', 'histogram') # same class multiple instance

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
        
        self.objectlocalizer     = ObjectLocalizer(self.objectlocalizer_cfg)
        self.objectdetector      = ObjectDetector(self.objectdetector_cfg)
        print("Object Detector is set")

    def image_info_callback(self, camera_info):
        h = camera_info.height
        w = camera_info.width
        K = np.array(camera_info.K, dtype=np.float64).reshape(3,3)

        self.imagehandler.set_cameraparams(K, [w,h])
        rospy.sleep(self.camera_info_callback_sleep)

    def tag_callback(self, data):
        
        for tag in data.detections:
            # print("Tag ",tag.id[0]," has been seen." )
            transform = self.tf_buffer.lookup_transform('map',
                                        'tag_' + str(tag.id[0]), #source frame
                                        rospy.Time(0),
                                        rospy.Duration(1)) #get the tf at first available time) #wait for 5 second

            xyz = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z ])
            self.tag_pos = np.vstack([self.tag_pos, xyz])
            self.mean_tag_pos = np.mean(self.tag_pos, axis=0)


    def run(self):
        def callback(image_msg, lidar_msg):
            # If Image Message is not empty 
            if image_msg.height != 0:               
                # transform the image msg to numpy array
                cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg)

                # transform the pointcloud msg to numpy array and remove nans
                point_cloud_XYZ = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar_msg)

                # translate and project PointCloud onto the Image
                point_cloud_XYZ = self.imagehandler.translatePoints(point_cloud_XYZ)
                img_pts, indices = self.imagehandler.projectPointsOnImage(point_cloud_XYZ)
                
                # 3D Lidar points that are inside the image frame  
                point_cloud_XYZ = point_cloud_XYZ[indices]

                # Detect objects in image
                result = self.objectdetector.detect(cv_image, return_image=True)

                # If no multiple instance for every class is allowed,
                # pick the one with highest confidance for every class.
                if not self.multiple_instance:
                    detected_objects = []
                    row_to_delete    = []
                    for i in range(len(result[0])):
                        if result[0]['class'][i] in detected_objects:
                            row_to_delete.append(i)
                        else:
                            detected_objects.append(result[0]['class'][i])
                    
                    result[0] = result[0].drop(row_to_delete, axis=0)

                # Visualize all Lidar points that are inside the image frame
                if self.visualize and self.visualize_all_points:
                    for idx, pt in enumerate(img_pts):
                        dist = np.linalg.norm(point_cloud_XYZ[idx])
                        color = depth_color(dist)
                        cv2.circle(result[1], tuple(pt), 1, color)


                object_poses, indices_list = self.objectlocalizer.localize(result[0], img_pts,  point_cloud_XYZ)

                # For every detected image object
                for i in range(len(result[0])):
                    
                    xyz = object_poses[i]

                    if np.any(np.isnan(xyz)):
                        continue

                    # transformstamped_("blackfly_right_optical_link", "object", xyz, [0,0,0,1])
                    self.TF_br.sendTransform(transformstamped_("blackfly_right_optical_link", "object", xyz, [0,0,0,1]))
                    
                    transform = self.tf_buffer.lookup_transform("map", "object",
                                                                rospy.Time(0),
                                                                rospy.Duration(10.0))
                    
                    obj_in_map = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z ])
                    markers = MarkerArray()
                    markers.markers.append(marker_("object", self.obj_id, obj_in_map, [1.0,0,0]))
                    if self.mean_tag_pos is not None:
                        markers.markers.append(marker_("tag", 1, self.mean_tag_pos, [0,1.0,0]))
                    self.marker_pub.publish(markers)                

                    self.obj_id += 1
                    in_BB_XYZ = point_cloud_XYZ[indices_list[i]]
                    # Visualize pointcloud on the image with objects
                    if self.visualize and not self.visualize_all_points:
                        for idx, pt in enumerate(img_pts[indices_list[i]]):
                            dist = np.linalg.norm(in_BB_XYZ[idx])
                            color = depth_color(dist)
                            cv2.circle(result[1], tuple(pt), 1, color)
                img_msg = self.cv_bridge.cv2_to_imgmsg(result[1], 'bgr8')
                img_msg.header.frame_id = "blackfly_right_optical_link"
                self.out_image_pub.publish(img_msg)

        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic , CameraInfo, self.image_info_callback)
        self.tag_sub         = rospy.Subscriber(self.tag_topic , AprilTagDetectionArray, self.tag_callback) 
        self.synchronizer.registerCallback(callback)
        rospy.spin()

if __name__ == '__main__':

    node = Node()
    print("Detection started")
    node.run()
    
