#!/usr/bin/env python3

import rospy
from object_detection_msgs.msg import ObjectDetectionInfoArray
import csv
import os
import math

class ObjectDetectionLogger:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('object_detection_logger', anonymous=True)
        
        # Create a subscriber to the /object_detector/detection_info topic
        self.subscriber = rospy.Subscriber('/object_detector/detection_info', ObjectDetectionInfoArray, self.callback)
        
        # File to store the detections
        self.file_path = os.path.join(os.path.expanduser('~'), 'detection_info.csv')
        
        # Open the file in write mode and set up the CSV writer
        self.file = open(self.file_path, mode='w')
        self.csv_writer = csv.writer(self.file)
        
        # Write the header row
        self.csv_writer.writerow(['Object ID', 'Class ID', 'Confidence', 'Position X', 'Position Y', 'Position Z'])
        
        # List to store logged detections
        self.logged_detections = []

    def is_within_radius(self, pos1, pos2, radius=2.0):
        # Calculate the Euclidean distance between two points
        distance = math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)
        return distance <= radius

    def callback(self, data):
        # Callback function to process the incoming ObjectDetectionInfoArray message
        for detection in data.info:
            position = detection.position
            is_duplicate = False
            
            # Check if an object of the same class type is already logged within the specified radius
            for logged in self.logged_detections:
                if (logged['class_id'] == detection.class_id and 
                    self.is_within_radius(logged['position'], position)):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                rospy.loginfo(f"Object {detection.id}: Class ID: {detection.class_id}, Confidence: {detection.confidence}, Position (x: {position.x}, y: {position.y}, z: {position.z})")
                
                # Print the detection info to the console
                print(f"Detected Object - ID: {detection.id}, Class ID: {detection.class_id}, Confidence: {detection.confidence}, Position: ({position.x}, {position.y}, {position.z})")
                
                # Write the detection info to the CSV file
                self.csv_writer.writerow([detection.id, detection.class_id, detection.confidence, position.x, position.y, position.z])
                
                # Add the detection to the logged list
                self.logged_detections.append({'class_id': detection.class_id, 'position': position})

    def run(self):
        # Keep the node running until it is shut down
        rospy.spin()
        
        # Close the file when the node is shut down
        self.file.close()

if __name__ == '__main__':
    try:
        logger = ObjectDetectionLogger()
        logger.run()
    except rospy.ROSInterruptException:
        pass
