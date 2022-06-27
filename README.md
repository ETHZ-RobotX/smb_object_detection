# Multimodal Object Detection and Mapping

This package enables detection and localization of objects using camera and lidar inputs. Object detections in 2D images are mapped to lidar scans to retreive the 3D object postions to scale.

For the object detection, the package utilizes [yolov5](https://github.com/ultralytics/yolov5).

## Install
The full pipeline has been written with Python3. The necessary libraries can be found in `requirements.txt`. They should be installed in an environment which can be reached by ROS. Furthermore, make sure that all ros package dependencies are met.

In the source directory of object_detection, execute:

```
rosdep install --from-paths . --ignore-src --os=ubuntu:focal -r -y
pip install -r requirements.txt
```

### Loading Detection Model From Local Repository
By default, a pretrained yolov5 model is used for object detection. This model can be loaded from an online repository, however, it is often convenient to load the model from a local directory. To do so, download a model from the [release page](https://github.com/ultralytics/yolov5/releases)

Then ensure that the ROS parameter `model_path` points to the directory where the model is stored and that `model` reflects the name of the model, e.g. `yolov5l6`.  For more information on the different available launch parameters see [below](#launching-and-parameters)

## Launching and Parameters

The detection and localization pipeline can be launched using

```
roslaunch object_detection object_detection.launch
```

The default setup assumes lauching the pipeline on a remote NVIDIA Jetson. For a different setup, e.g. launching on a local mahcine, inspect the different launch parameters below.

| Flag                         | Default                                               | Description                    |
| ---------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| smb_name                     | $(optenv SMB_NAME smb261)                             | The name of the SMB in the format smb26x. Only relevant for loading projector configuration. |
| gpu                          | remote                                                | Whether to utilize GPU. Can be set 'local' to run on the GPU of the local machine (this may require additional setup of CUDA libraries), 'remote' to run on the NVIDIA Jetson GPU, or 'off' to run on the CPU of the local machine. For debugging, 'off' is the best option. |
| GPU_user                     | $(env USER)                                           | Username to use on the NVIDIA Jetson GPU |
| debayer_image                | true                                                  | Whether to debayer images. If this is set to false, `camera_topic` and `camera_info_topic` probably needs to be changed. |
| input_camera_name            | /versavis/cam0/slow                                   | Input camera name. The exact topics needed are inferred based on the camera name. By default refers to a temporally downsampled image stream.|
| camera_topic                 | $(arg input_camera_name)/image_color                  | The topic of the images to detect objects in. |
| camera_info_topic            | $(arg input_camera_name)/camera_info                  | The topic of the camera info, containing the camera calibration parameters. Used once during initialization. |
| lidar_topic                  | /rslidar_points                                       | Lidar mesage topic name                |
| camera_lidar_sync_queue_size | 10                                                    | How many frame should be searched to find appropiate time stamp. Please refer [here](http://wiki.ros.org/message_filters#ApproximateTime_Policy).|
| camera_lidar_sync_slop       | 0.1                                                   | The maximim time difference between syncronized topics. In real-time usage it might be large due to the fact that sensors start running at different time. Please refer [here](http://wiki.ros.org/message_filters#ApproximateTime_Policy).|
| model_path                   | /usr/share/yolo/models                                | The path to the directory containing the local yolov5 model. If left empty, `''`, the model will be loaded from the online repository.|
| model                        | yolov5l6                                              | The yolo model that will be used. Internet connection might be needed to install the weights. For available models please refer [here](https://github.com/ultralytics/yolov5/releases). The model size is the main factor of the speed of the package. Smaller models can be utilized for realtime usage such as yolov5n / yolov5n6. |
| device                       | cpu                                                   | The device that the model will run on. cpu or a cuda device as 0,1,2 ... By default this is set automatically for smb usage based on the `gpu` argument.|
| confident                    | 0.4                                                   | Non-maxima supression threshold, used in the elimination of duplicate detections.|
| iou                          | 0.1                                                   | If iou of two detections is more than this threshold, the one with the lower confident score will be discarded.|
| object_detection_classes     | 0                                                     | Comma-separated list defining which objects should be detected. The association between object name and number can be found [here](https://github.com/ethz-asl/darknet_catkin/blob/master/data/coco.names) (the numbering starts from 0).|
| multiple_instance            | False                                                 | If it is False, only one instance per class will be detected. Multiple detections can be visible in the output detection image, but only the instance with highest confidence score will be localized in the point cloud. Leave as false if you only expect one instance of the same instance in the scene at once.|
| model_method | hdbscan | The model method to determine which points are on the object. If the object is solid (no gap on its body), `center` can be also used. |
| min_cluster_size | 5 | Minimum number of lidar points that fall into the bounding box of an object for accurate measurement. It should be greater or equal to 3 for the model_method `hdbscan`; for the `center`, it can be even 1. The objects at the distance might not have enough points on them, therefore cannot be localized.|
| ground_percentage | 25 | Starting from the ground level till the camera center, the percentage of height that is considered as ground and therefore discarded. |
| bb_contract_percentage | 10 | The percentage how much the edges of bounding box should be contracted. |

Apart from the above parameters, the following parameters are available in `output_params.yaml` to configure the output of the node.
| Flag                         | Default                                               | Description                   |
| ------------------------------ | ------------------------------------ | -------------------------------------------- |
| project_object_points_to_image | true                                 | Whether or not to project and display the lidar points associated with detected objects into the detection image |
| project_all_points_to_image    | false                                | Whether or not to project all lidar points visible to the camera into the detection image. This is mainly useful for debugging purposes. |
| object_detection_pos_topic     | ~object_positions                    | The topic on which the object poses are published as a PoseArray message. |
| object_detection_output_image_topic | ~detections_in_image            | The topic on which the detection image (input image with overlayed bounding boxes for detected objects) is published. |
| object_detection_point_clouds_topic | ~detection_point_clouds         | The topic on which the point clouds associated to each object are published. (Published as an array of PointCloud2 messages) |
| object_detection_info_topic         | ~detection_info                 | The topic on which all other potentially important detection information is published. See the [message definition](object_detection_msgs/msg/ObjectDetectionInfo.msg) for details. |

### Published topics

- *detection_info*
    - Information related to each detection, such as class name, confidence, etc.
    - Published as an [array](object_detection_msgs/msg/ObjectDetectionInfoArray.msg) of [ObjectDetectionInfo](object_detection_msgs/msg/ObjectDetectionInfo.msg)
- *detection_point_clouds*
    - The lidar points associated to each detection.
    - Published as an [array](object_detection_msgs/msg/PointCloudArray.msg) of PointCloud2 messages
- *detections_in_image*
    - The input image with detections marked by bounding boxes. Additionally with optional lidar points projected.
- *object_poses*
    - The poses of each detected object.
    - Published as a PoseArray for easy visualisation in RViz

### **IMPORTANT REMARK ABOUT THE OUTPUT** ###
If the z axis of a detected object is -1, the object is not correctly localized. This happens if an object is detected by the camera but there are not enough Lidar points/measurement inside the bounding box of it.

## Launch Using Rosbag
The launch folder contains a launch file for testing the pipline based on a provided rosbag. Apart from launching the detection node and playing the rosbag, it also launches an RViz window with useful visualisation of relevant topics. Currently the existing rosbag has slightly different outputs than on the newest setup of the camera. To compensate for this, the rosbag launch file contains a topic dropper node to create the `/versavis/cam0/slow` topics.
