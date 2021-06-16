import argparse
import utils

# rospy for the subscriber
import rospy

# ROS message
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
from std_msgs.msg import String

# ROS Image message -> OpenCV2 image converter
import cv2
import base64
from cv_bridge import CvBridge

# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)
from keras.models import load_model

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K

class RosTensorFlow():
    def __init__(self, model, image_topic):
        self.epsilon = 1
        self.graph = tf.compat.v1.get_default_graph()

        self.model = load_model(model)

        # Right Attack (counter clockwise)
        self.loss_right = K.mean(-self.model.output, axis=-1)
        self.grads_right = K.gradients(self.loss_right, self.model.input)
        self.delta_right = K.sign(self.grads_right[0])

        # Left Attack (clockwise)
        self.loss_left = K.mean(self.model.output, axis=-1)
        self.grads_left = K.gradients(self.loss_left, self.model.input)
        self.delta_left = K.sign(self.grads_left[0])

        self.sess = tf.compat.v1.keras.backend.get_session()

        self._cv_bridge = CvBridge()

        # Input Image
        self._sub = rospy.Subscriber(image_topic, Image, self.input_callback, queue_size=1)
        # Output steering angle
        self._pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Activate attacks based on the topic
        self._attack_sub = rospy.Subscriber('/attack', Int32, self.attack_callback, queue_size=1)
        self.attack = 0

        # Publish images to the web UI
        self.raw_pub = rospy.Publisher('/raw_img', String, queue_size=1)
        self.input_pub = rospy.Publisher('/input_img', String, queue_size=1)
        self.perturb_pub = rospy.Publisher('/perturb_img', String, queue_size=1)
        self.adv_pub = rospy.Publisher('/adv_img', String, queue_size=1)

    def publish_image(self, cv_image, pub_topic):
        _, buffer = cv2.imencode('.jpg', cv_image)
        image_as_str = base64.b64encode(buffer).decode('utf-8')
        pub_topic.publish(image_as_str)

    def attack_callback(self, attack_msg):
        self.attack = attack_msg.data
        print('Attack Type:', self.attack)

    def input_callback(self, input_raw_image_msg):
        input_raw_image_msg = self._cv_bridge.imgmsg_to_cv2(input_raw_image_msg, "bgr8")
        input_raw_image_msg = cv2.resize(input_raw_image_msg, (320, 160), interpolation = cv2.INTER_AREA)

        # Publish the raw image
        self.publish_image(input_raw_image_msg, self.raw_pub)

        # apply the preprocessing
        input_cv_image = utils.preprocess(input_raw_image_msg) 
        
        # Publish the model input image
        self.publish_image(input_cv_image, self.input_pub)
        
        with self.graph.as_default():

            if self.attack > 0:
                # 1 --> Left Attack
                if self.attack == 1:
                    perturb = self.epsilon * self.sess.run(self.delta_left, feed_dict={self.model.input:np.array([input_cv_image])})
                # 2 --> Right Attack
                if self.attack == 2:
                    perturb = self.epsilon * self.sess.run(self.delta_right, feed_dict={self.model.input:np.array([input_cv_image])})

                perturb = perturb.reshape(160, 320, 3)

                # Publish the perturbation
                self.publish_image(perturb, self.perturb_pub)

                # Publish the adversarial image
                self.publish_image(perturb + input_cv_image, self.adv_pub)

                angle = self.epsilon * self.sess.run(self.model.output, feed_dict={self.model.input:np.array([perturb + input_cv_image])})
                no_attack_angle = self.sess.run(self.model.output, feed_dict={self.model.input:np.array([input_cv_image])})
                print('{0} --> {1}'.format(no_attack_angle[0][0], angle[0][0]))
            else:
                angle = self.sess.run(self.model.output, feed_dict={self.model.input:np.array([input_cv_image])})
                print(angle[0][0])

            msg = Twist()
            msg.linear.x = 0.2
            msg.linear.y = 0
            msg.linear.z = 0
            msg.angular.z = angle[0][0]
            self._pub.publish(msg)


    def main(self):
        rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Line Following')
    parser.add_argument('--env', help='environment', choices=['gazebo', 'turtlebot'], type=str, required=True)
    parser.add_argument('--model', help='deep learning model', type=str, required=True)
    args = parser.parse_args()

    rospy.init_node('ros_line_track_tf')

    if args.env == 'gazebo':
        image_topic = "/camera/rgb/image_raw"
    if args.env == 'turtlebot':       
        image_topic = "/raspicam_node/image_raw"

    tensor = RosTensorFlow(args.model, image_topic);
    tensor.main()
