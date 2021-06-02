# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge

import cv2

# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)
from keras.models import load_model

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K


class RosTensorFlow():
    def __init__(self):
        self.epsilon = 1
        self.graph = tf.compat.v1.get_default_graph()

        self.model = load_model("model.h5")
        self.loss = K.mean(self.model.output, axis=-1)
        self.grads = K.gradients(self.loss, self.model.input)
        self.delta = K.sign(self.grads[0])

        self.sess = tf.compat.v1.keras.backend.get_session()

        self._cv_bridge = CvBridge()

        self._sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)


    def callback(self, image_msg):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        cv_image = cv2.resize(cv_image, (320, 160), interpolation = cv2.INTER_AREA)
        with self.graph.as_default():
            noise = self.epsilon * self.sess.run(self.delta, feed_dict={self.model.input:np.array([cv_image])})
            noise = noise.reshape(160, 320, 3)
            no_angle = self.epsilon * self.sess.run(self.model.output, feed_dict={self.model.input:np.array([cv_image])})
            angle = self.epsilon * self.sess.run(self.model.output, feed_dict={self.model.input:np.array([cv_image+noise])})
            # cv_image = cv_image + noise
            print(no_angle[0][0])
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
    rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()