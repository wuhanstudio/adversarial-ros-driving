#! /usr/bin/python

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import pandas as pd
from datetime import datetime
import os.path

steering_angle = 0
speed = 0

# Instantiate CvBridge
bridge = CvBridge()

column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

if os.path.isfile("driving_log.csv"):
    df = pd.read_csv("driving_log.csv")
else:
    df = pd.DataFrame(columns=column_names)

def cmd_callback(msg):
    global steering_angle, speed
    steering_angle = msg.angular.z
    speed = msg.linear.x

def image_callback(msg):
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S-%f")
    try:
        global df
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2_img = cv2.resize(cv2_img, (320, 160), interpolation = cv2.INTER_AREA)
        filename = 'IMG/' + current_time + '.jpg'
        cv2.imwrite(filename, cv2_img)
        df = df.append({
            'center': filename,
            'left': filename,
            'right': filename,
            'steering': steering_angle,
            'throttle': 0,
            'brake': 0,
            'speed': speed
        }, ignore_index=True)
    except CvBridgeError as e:
        print(e)

def myhook():
    print("shutdown time!")
    df.to_csv('driving_log.csv', index=False)

def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/camera/rgb/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)

    cmd_topic = "/cmd_vel"
    rospy.Subscriber(cmd_topic, Twist, cmd_callback)

    rospy.on_shutdown(myhook)

    print('Collecting Center Data')

    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
