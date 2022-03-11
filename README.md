# Multimodal Object Detection and Mapping
Detection and mapping of objects with camera and lidar

The main branch contains the necessary packages to run the pipeline with the packages of SMB. The branch `stand_alone` can run rosbags and visualize them without needing any additional package. 

## Install
The full pipeline has been written with Python3. The necessary libraries can be found in requirements.txt . They should be installed in an environment which can be reached by ROS. 

pip install -r requirements.txt

## Start Multimodal Object Detection and Mapping
Adapt the smb name in the launch file below to the correct SMB number and run to start the object mapping node.
```
roslaunch object_detection object_detection.launch
```
Detections will be published in the optical frame, which is a parameter to the system. Other parameters such as the ones listed below can also be added and adapted in the launch file.

| Flag | Default | Description |
| --- | --- | --- |
| camera_calib_path | $(find object_detection)/cfg/smb264_camera_model.yaml | The path of the yaml file that contains the parameters of the camera |
| optical_frame_name | blackfly_right_optical_link | Camera frame name |
| input_camera_name | /versavis/cam0 | Input camera name/topic  |
| lidar_topic | /rslidar_points | Lidar mesage topic name |
| object_detection_topic | /objects | topic name of the detection output|
| camera_lidar_sync_queue_size | 10 | how many frame should be searched to find appropiate time stamp. Please refer [here](http://wiki.ros.org/message_filters#ApproximateTime_Policy) |
| camera_lidar_sync_slop | 0.1 | the maximim time difference between syncronized topics. In real-time usage it might be large due to the fact that sensors start running at different time. Please refer  [here](http://wiki.ros.org/message_filters#ApproximateTime_Policy)|
| model | yolov5l6 | the yolo model that will be used. Internet connection might be needed to install the weights. For available models please refer [here](https://github.com/ultralytics/yolov5/releases) |
| device | cpu | The device that the model will run on. cpu or a cuda device as 0,1,2 ... |
| confident | 0.4 | Non maxima supression threshold, used in the elimination of duplicate detections |
| iou | 0.1 | If iou of two detections is more than this threshold, the one with lower confident will be discarded. |
| darknet_nms_threshold | 0.45 | Non maxima supression threshold, used in the elimination of duplicate detections |
| object_detection_classes | 0 | Comma separated list. The association between object name and number can be found [here](https://github.com/ethz-asl/darknet_catkin/blob/master/data/coco.names) (the numbering starts from 0) |
| multiple_instance | False | If it is False, only one instance per class will be detected. Only the instance with highest instance will be considered. It is better have False to prevent wrong detections. |
| model_method | hdbscan | The model method to which points are on the object. If the object is solid (no gap on its body), `center` can be also used.  |
| ground_percentage | 25 | Starting from the ground level till the camera center, the percentage of height that is considered as ground and therefore discarded. |
| bb_contract_percentage | 10 | the percentage how much the edges of bounding box should be contracted.  |
| verbose| False | If it is True the output message topic contains the optional topics|


## Extra Package: Object Visualization
This package has only one script that shows how the object detection output should be used. This package can be also used to visualize the detections on image and also on the map frame with markers. 