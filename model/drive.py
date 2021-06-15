# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge
import utils
import matplotlib.pyplot as plt

import cv2

import base64

from std_msgs.msg import String

# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)
from keras.models import load_model

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K

import argparse

class RosTensorFlow():
    def __init__(self):
        self.epsilon = 1
        self.graph = tf.compat.v1.get_default_graph()

        self.model = load_model("model.h5")

        self.loss_right = K.mean(-self.model.output, axis=-1)
        self.grads_right = K.gradients(self.loss_right, self.model.input)
        self.delta_right = K.sign(self.grads_right[0])

        self.loss_left = K.mean(self.model.output, axis=-1)
        self.grads_left = K.gradients(self.loss_left, self.model.input)
        self.delta_left = K.sign(self.grads_left[0])

        self.sess = tf.compat.v1.keras.backend.get_session()

        self._cv_bridge = CvBridge()

        self._sub = None
        self._pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self._attack_sub = rospy.Subscriber('/attack', Int32, self.attack_callback, queue_size=1)
        self.attack = 0

        self.raw_pub = rospy.Publisher('/raw_img', String, queue_size=10)
        self.input_pub = rospy.Publisher('/input_img', String, queue_size=10)
        self.perturb_pub = rospy.Publisher('/perturb_img', String, queue_size=10)
        self.adv_pub = rospy.Publisher('/adv_img', String, queue_size=10)

    def attack_callback(self, attack_msg):
        self.attack = attack_msg.data
        print(self.attack)

    def callback(self, image_msg):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        cv_image = cv2.resize(cv_image, (320, 160), interpolation = cv2.INTER_AREA)

        _, buffer = cv2.imencode('.jpg', cv_image)
        image_as_str = base64.b64encode(buffer).decode('utf-8')
        self.raw_pub.publish(image_as_str)

        cv_image = utils.preprocess(cv_image) # apply the preprocessing
        _, buffer = cv2.imencode('.jpg', cv_image)
        image_as_str = base64.b64encode(buffer).decode('utf-8')
        self.input_pub.publish(image_as_str)
        # plt.imshow(cv_image)
        # plt.show()
        
        with self.graph.as_default():

            if self.attack > 0:
                if self.attack == 1:
                    noise = self.epsilon * self.sess.run(self.delta_left, feed_dict={self.model.input:np.array([cv_image])})
                if self.attack == 2:
                    noise = self.epsilon * self.sess.run(self.delta_right, feed_dict={self.model.input:np.array([cv_image])})

                noise = noise.reshape(160, 320, 3)

                _, buffer = cv2.imencode('.jpg', noise)
                image_as_str = base64.b64encode(buffer).decode('utf-8')
                self.perturb_pub.publish(image_as_str)

                _, buffer = cv2.imencode('.jpg', noise + cv_image)
                image_as_str = base64.b64encode(buffer).decode('utf-8')
                self.adv_pub.publish(image_as_str)

                angle = self.epsilon * self.sess.run(self.model.output, feed_dict={self.model.input:np.array([cv_image+noise])})
                no_angle = self.sess.run(self.model.output, feed_dict={self.model.input:np.array([cv_image])})
                print('{0} --> {1}'.format(no_angle[0][0], angle[0][0]))
            else:
                angle = self.sess.run(self.model.output, feed_dict={self.model.input:np.array([cv_image])})
            # print('{0} --> {1}'.format(no_angle[0][0], angle[0][0]))

            msg = Twist()
            msg.linear.x = 0.02
            msg.linear.y = 0
            msg.linear.z = 0
            msg.angular.z = angle[0][0]
            self._pub.publish(msg)


    def main(self):
        rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Line Following')
    parser.add_argument('--env', help='environment', choices=['gazebo', 'turtlebot'], type=str, required=True)
    args = parser.parse_args()

    rospy.init_node('ros_tensorflow')

    if args.env == 'gazebo':
        image_topic = "/camera/rgb/image_raw"
    if args.env == 'turtlebot':       
        image_topic = "/raspicam_node/image_raw"

    tensor = RosTensorFlow()
    tensor._sub = rospy.Subscriber(image_topic, Image, tensor.callback, queue_size=1)
    tensor.main()
