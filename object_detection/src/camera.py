"""Publish a video as ROS messages.
"""

import argparse

import numpy as np

import cv2

import rospy

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo



from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

def main():
    """Publish a video as ROS messages.
    """
    # Patse arguments.

    # Set up node.
    rospy.init_node("video_publisher", anonymous=True)
    img_pub = rospy.Publisher("/versavis/cam0/image_raw", Image,
                              queue_size=1)


    # Open video.
    video = cv2.VideoCapture(0)
    rate = rospy.Rate(20) # 10hz
    # Loop through video frames.
    while not rospy.is_shutdown() and video.grab():
        tmp, img = video.retrieve()

        # cv2.imshow(" asd ", img)
        # cv2.waitKey(1)
        # Publish image.
        img_msg = bridge.cv2_to_imgmsg(img)
        img_msg.header.stamp = rospy.Time.now()
        img_msg.encoding = 'bgr8'

        img_pub.publish(img_msg)

        rate.sleep()

    video.release()
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass