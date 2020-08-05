#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import QuaternionStamped
from sensor_msgs.msg import Imu
from tf.transformations import quaternion_from_euler

from cv_bridge import CvBridge, CvBridgeError

import cv2
from PIL import Image
import math

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

import vggbased_network

class AttitudeEstimation:
    def __init__(self, frame_id, device, size, mean, std, net):
        ## subscriber
        self.sub_imgae = rospy.Subscriber("/image_raw", ImageMsg, self.callbackImage, queue_size=1, buff_size=2**24)
        ## publisher
        self.pub_vector = rospy.Publisher("/dnn/g_vector", Vector3Stamped, queue_size=1)
        self.pub_quat = rospy.Publisher("/dnn/attitude", QuaternionStamped, queue_size=1)
        self.pub_accel = rospy.Publisher("/dnn/g_vector_with_cov", Imu, queue_size=1)
        ## msg
        self.v_msg = Vector3Stamped()
        self.q_msg = QuaternionStamped()
        self.accel_msg = Imu()
        ## cv_bridge
        self.bridge = CvBridge()
        ## copy arguments
        self.frame_id = frame_id
        self.device = device
        self.img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.net = net

    def callbackImage(self, msg):
        print("----------")
        start_clock = rospy.get_time()
        try:
            img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("img_cv.shape = ", img_cv.shape)
            outputs = self.dnnPrediction(img_cv)
            self.inputMsg(outputs)
            self.publication(msg.header.stamp)
        except CvBridgeError as e:
            print(e)
        print("Period [s]: ", rospy.get_time() - start_clock, "Frequency [hz]: ", 1/(rospy.get_time() - start_clock))

    def dnnPrediction(self, img_cv):
        img_pil = self.cvToPIL(img_cv)
        img_transformed = self.img_transform(img_pil)
        inputs = img_transformed.unsqueeze_(0)
        inputs = inputs.to(self.device)
        outputs = self.net(inputs)
        print("outputs = ", outputs)
        return outputs

    def cvToPIL(self, img_cv):
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv)
        return img_pil

    def inputMsg(self, outputs):
        ## get covariance matrix
        cov = self.getCovMatrix(outputs)
        ## tensor to numpy
        outputs = outputs[0].detach().numpy()
        cov = cov[0].detach().numpy()
        ## Vector3Stamped
        self.v_msg.vector.x = -outputs[0]
        self.v_msg.vector.y = -outputs[1]
        self.v_msg.vector.z = -outputs[2]
        ## QuaternionStamped
        r = math.atan2(outputs[1], outputs[2])
        p = math.atan2(-outputs[0], math.sqrt(outputs[1]*outputs[1] + outputs[2]*outputs[2]))
        y = 0.0
        q_tf = quaternion_from_euler(r, p, y)
        self.q_msg.quaternion.x = q_tf[0]
        self.q_msg.quaternion.y = q_tf[1]
        self.q_msg.quaternion.z = q_tf[2]
        self.q_msg.quaternion.w = q_tf[3]
        ## Imu
        self.inputNanToImuMsg(self.accel_msg)
        self.accel_msg.linear_acceleration.x = -outputs[0]
        self.accel_msg.linear_acceleration.y = -outputs[1]
        self.accel_msg.linear_acceleration.z = -outputs[2]
        for i in range(cov.size):
            self.accel_msg.linear_acceleration_covariance[i] = cov[i//3, i%3]
        ## print
        print("r = ", r, ", p = ", p)
        print("cov = ", cov)

    def getCovMatrix(self, outputs):
        L = self.getTriangularMatrix(outputs)
        Ltrans = torch.transpose(L, 1, 2)
        LL = torch.bmm(L, Ltrans)
        return LL

    def getTriangularMatrix(self, outputs):
        elements = outputs[:, 3:9]
        L = torch.zeros(outputs.size(0), elements.size(1)//2, elements.size(1)//2)
        L[:, 0, 0] = torch.exp(elements[:, 0])
        L[:, 1, 0] = elements[:, 1]
        L[:, 1, 1] = torch.exp(elements[:, 2])
        L[:, 2, 0] = elements[:, 3]
        L[:, 2, 1] = elements[:, 4]
        L[:, 2, 2] = torch.exp(elements[:, 5])
        return L

    def inputNanToImuMsg(self, imu):
        imu.orientation.x = math.nan
        imu.orientation.y = math.nan
        imu.orientation.z = math.nan
        imu.orientation.w = math.nan
        imu.angular_velocity.x = math.nan
        imu.angular_velocity.y = math.nan
        imu.angular_velocity.z = math.nan
        imu.linear_acceleration.x = math.nan
        imu.linear_acceleration.y = math.nan
        imu.linear_acceleration.z = math.nan
        for i in range(len(imu.linear_acceleration_covariance)):
            imu.orientation_covariance[i] = math.nan
            imu.angular_velocity_covariance[i] = math.nan
            imu.linear_acceleration_covariance[i] = math.nan

    def publication(self, stamp):
        print("delay[s]: ", (rospy.Time.now() - stamp).to_sec())
        ## Vector3Stamped
        self.v_msg.header.stamp = stamp
        self.v_msg.header.frame_id = self.frame_id
        self.pub_vector.publish(self.v_msg)
        ## QuaternionStamped
        self.q_msg.header.stamp = stamp
        self.q_msg.header.frame_id = self.frame_id
        self.pub_quat.publish(self.q_msg)
        ## Imu
        self.accel_msg.header.stamp = stamp
        self.accel_msg.header.frame_id = self.frame_id
        self.pub_accel.publish(self.accel_msg)

def main():
    ## Node
    rospy.init_node('attitude_estimation', anonymous=True)
    ## Param
    weights_path = rospy.get_param("/weights_path", "../weights/weights.pth")
    print("weights_path = ", weights_path)
    frame_id = rospy.get_param("/frame_id", "/base_link")
    print("frame_id = ", frame_id)

    ## Device check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = ", device)
    # print("torch.cuda.current_device() = ", torch.cuda.current_device())
    ## size, mean, std
    size = 224  #VGG16
    mean = ([0.5, 0.5, 0.5])
    std = ([0.5, 0.5, 0.5])
    ## Network
    net = vggbased_network.OriginalNet()
    print(net)
    net.to(device)
    ## Load weights
    weights_was_saved_in_same_device = False
    if weights_was_saved_in_same_device:
        ## saved in CPU -> load in CPU, saved in GPU -> load in GPU
        load_weights = torch.load(weights_path)
    else:
        ## saved in GPU -> load in CPU
        load_weights = torch.load(weights_path, map_location={"cuda:0": "cpu"})
    net.load_state_dict(load_weights)
    ## set as eval
    net.eval()

    attitude_estimation = AttitudeEstimation(frame_id, device, size, mean, std, net)

    rospy.spin()

if __name__ == '__main__':
    main()
