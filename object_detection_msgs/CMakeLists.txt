cmake_minimum_required(VERSION 3.0.2)
project(object_detection_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

add_message_files(
  DIRECTORY msg
  FILES
  ObjectDetection.msg
  ObjectDetectionArray.msg
  ObjectPosition.msg
  ObjectPositionArray.msg
  ObjectDetectionInfo.msg
  ObjectDetectionInfoArray.msg
  PointCloudArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  std_msgs
  geometry_msgs
  sensor_msgs
)