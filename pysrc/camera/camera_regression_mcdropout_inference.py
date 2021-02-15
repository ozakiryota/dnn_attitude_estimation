#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu

from cv_bridge import CvBridge, CvBridgeError

import cv2
from PIL import Image
import math
import numpy as np

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

import network_mod

class AttitudeEstimation:
    def __init__(self):
        print("--- mle_prediction ---")
        ## parameter-msg
        self.frame_id = rospy.get_param("/frame_id", "/base_link")
        print("self.frame_id = ", self.frame_id)
        ## parameter-dnn
        weights_path = rospy.get_param("/weights_path", "../../weights/weights.pth")
        print("weights_path = ", weights_path)
        resize = rospy.get_param("/resize", 224)
        print("resize = ", resize)
        mean_element = rospy.get_param("/mean_element", 0.5)
        print("mean_element = ", mean_element)
        std_element = rospy.get_param("/std_element", 0.5)
        print("std_element = ", std_element)
        self.num_mcsampling = rospy.get_param("/num_mcsampling", 25)
        print("self.num_mcsampling = ", self.num_mcsampling)
        ## subscriber
        self.sub_color_img = rospy.Subscriber("/color_image", ImageMsg, self.callbackColorImage, queue_size=1, buff_size=2**24)
        ## publisher
        self.pub_vector = rospy.Publisher("/dnn/g_vector", Vector3Stamped, queue_size=1)
        self.pub_accel = rospy.Publisher("/dnn/g_vector_with_cov", Imu, queue_size=1)
        ## msg
        self.v_msg = Vector3Stamped()
        self.accel_msg = Imu()
        ## cv
        self.bridge = CvBridge()
        self.color_img_cv = np.empty(0)
        ## dnn
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device = ", self.device)
        self.img_transform = self.getImageTransform(resize, mean_element, std_element)
        self.net = self.getNetwork(resize, weights_path)
        self.enable_dropout()

    def getImageTransform(self, resize, mean_element, std_element):
        mean = ([mean_element, mean_element, mean_element])
        std = ([std_element, std_element, std_element])
        img_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return img_transform

    def getNetwork(self, resize, weights_path):
        net = network_mod.Network(resize, dim_fc_out=3, use_pretrained_vgg=False)
        print(net)
        net.to(self.device)
        net.eval()
        ## load
        if torch.cuda.is_available():
            loaded_weights = torch.load(weights_path)
            print("Loaded [GPU -> GPU]: ", weights_path)
        else:
            loaded_weights = torch.load(weights_path, map_location={"cuda:0": "cpu"})
            print("Loaded [GPU -> CPU]: ", weights_path)
        net.load_state_dict(loaded_weights)
        return net

    def enable_dropout(self):
        for module in self.net.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

    def callbackColorImage(self, msg):
        try:
            self.color_img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("msg.encoding = ", msg.encoding)
            print("self.color_img_cv.shape = ", self.color_img_cv.shape)
            outputs = self.dnnPrediction()
            self.inputToMsg(outputs)
            self.publication(msg.header.stamp)
        except CvBridgeError as e:
            print(e)

    def dnnPrediction(self):
        print("----------")
        start_clock = rospy.get_time()
        ## inference
        inputs_color = self.transformImage()
        print("inputs_color.size() = ", inputs_color.size())
        list_outputs = []
        for _ in range(self.num_mcsampling):
            outputs = self.net(inputs_color)
            list_outputs.append(outputs.cpu().detach().numpy()[0])
        # print("list_outputs = ", list_outputs)
        ## reset
        print("Period [s]: ", rospy.get_time() - start_clock)
        return list_outputs

    def transformImage(self):
        ## color
        color_img_pil = self.cvToPIL(self.color_img_cv)
        color_img_tensor = self.img_transform(color_img_pil)
        inputs_color = color_img_tensor.unsqueeze_(0)
        inputs_color = inputs_color.to(self.device)
        return inputs_color

    def cvToPIL(self, img_cv):
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv)
        return img_pil

    def inputToMsg(self, outputs):
        ## get mean and covariance matrix
        mean = np.array(outputs).mean(0)
        print("mean.shape = ", mean.shape)
        cov = self.getCovMatrix(outputs)
        ## Vector3Stamped
        self.v_msg.vector.x = -mean[0]
        self.v_msg.vector.y = -mean[1]
        self.v_msg.vector.z = -mean[2]
        ## Imu
        self.inputNanToImuMsg(self.accel_msg)
        self.accel_msg.linear_acceleration.x = -mean[0]
        self.accel_msg.linear_acceleration.y = -mean[1]
        self.accel_msg.linear_acceleration.z = -mean[2]
        for i in range(cov.size):
            self.accel_msg.linear_acceleration_covariance[i] = cov[i//3, i%3]
        ## print
        print("mean = ", mean)
        print("cov = ", cov)

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

    def getCovMatrix(self, outputs):
        cov = np.cov(outputs, rowvar=False, bias=True)
        return cov

    def publication(self, stamp):
        print("delay[s]: ", (rospy.Time.now() - stamp).to_sec())
        ## Vector3Stamped
        self.v_msg.header.stamp = stamp
        self.v_msg.header.frame_id = self.frame_id
        self.pub_vector.publish(self.v_msg)
        ## Imu
        self.accel_msg.header.stamp = stamp
        self.accel_msg.header.frame_id = self.frame_id
        self.pub_accel.publish(self.accel_msg)

def main():
    ## Node
    rospy.init_node('attitude_estimation', anonymous=True)

    attitude_estimation = AttitudeEstimation()

    rospy.spin()

if __name__ == '__main__':
    main()
