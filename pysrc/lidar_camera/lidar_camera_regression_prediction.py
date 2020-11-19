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
        weights_path = rospy.get_param("/weights_path", "../../weights/regression.pth")
        print("weights_path = ", weights_path)
        resize = rospy.get_param("/resize", 112)
        print("resize = ", resize)
        mean_element = rospy.get_param("/mean_element", 0.5)
        print("mean_element = ", mean_element)
        std_element = rospy.get_param("/std_element", 0.5)
        print("std_element = ", std_element)
        ## subscriber
        self.sub_color_img = rospy.Subscriber("/color_image", ImageMsg, self.callbackColorImage, queue_size=1, buff_size=2**24)
        self.sub_depth_img = rospy.Subscriber("/depth_image", ImageMsg, self.callbackDepthImage, queue_size=1, buff_size=2**24)
        ## publisher
        self.pub_vector = rospy.Publisher("/dnn/g_vector", Vector3Stamped, queue_size=1)
        ## msg
        self.v_msg = Vector3Stamped()
        ## cv
        self.bridge = CvBridge()
        self.color_img_cv = np.empty(0)
        self.depth_img_cv = np.empty(0)
        ## flag
        self.got_new_color_img = False
        self.got_new_depth_img = False
        ## dnn
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device = ", self.device)
        self.img_transform = self.getImageTransform(resize, mean_element, std_element)
        self.net = self.getNetwork(resize, weights_path)

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
        net = network_mod.Network(resize, dim_fc_out=3, use_pretrained=False)
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

    def callbackColorImage(self, msg):
        try:
            self.color_img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("msg.encoding = ", msg.encoding)
            print("self.color_img_cv.shape = ", self.color_img_cv.shape)
            self.got_new_color_img = True
            if self.got_new_color_img and self.got_new_depth_img:
                outputs = self.dnnPrediction()
                self.inputToMsg(outputs)
                self.publication(msg.header.stamp)
        except CvBridgeError as e:
            print(e)

    def callbackDepthImage(self, msg):
        try:
            self.depth_img_cv = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
            print("msg.encoding = ", msg.encoding)
            print("self.depth_img_cv.shape = ", self.depth_img_cv.shape)
            self.got_new_depth_img = True
            if self.got_new_color_img and self.got_new_depth_img:
                outputs = self.dnnPrediction()
                self.inputToMsg(outputs)
                self.publication(msg.header.stamp)
        except CvBridgeError as e:
            print(e)

    def dnnPrediction(self):
        print("----------")
        start_clock = rospy.get_time()
        ## inference
        inputs_color, inputs_depth = self.transformImage()
        print("inputs_color.size() = ", inputs_color.size())
        print("inputs_depth.size() = ", inputs_depth.size())
        outputs = self.net(inputs_color, inputs_depth)
        print("outputs = ", outputs)
        ## reset
        self.got_new_color_img = False
        self.got_new_depth_img = False
        print("Period [s]: ", rospy.get_time() - start_clock, "Frequency [hz]: ", 1/(rospy.get_time() - start_clock))
        return outputs

    def transformImage(self):
        ## color
        color_img_pil = self.cvToPIL(self.color_img_cv)
        color_img_tensor = self.img_transform(color_img_pil)
        inputs_color = color_img_tensor.unsqueeze_(0)
        inputs_color = inputs_color.to(self.device)
        ## depth
        depth_img_tensor = torch.from_numpy(self.depth_img_cv)
        inputs_depth = depth_img_tensor.unsqueeze_(0).unsqueeze_(0)
        inputs_depth = inputs_depth.to(self.device)
        return inputs_color, inputs_depth

    def cvToPIL(self, img_cv):
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv)
        return img_pil

    def inputToMsg(self, outputs):
        ## tensor to numpy
        outputs = outputs[0].cpu().detach().numpy()
        ## Vector3Stamped
        self.v_msg.vector.x = -outputs[0]
        self.v_msg.vector.y = -outputs[1]
        self.v_msg.vector.z = -outputs[2]

    def publication(self, stamp):
        print("delay[s]: ", (rospy.Time.now() - stamp).to_sec())
        ## Vector3Stamped
        self.v_msg.header.stamp = stamp
        self.v_msg.header.frame_id = self.frame_id
        self.pub_vector.publish(self.v_msg)

def main():
    ## Node
    rospy.init_node('attitude_estimation', anonymous=True)

    attitude_estimation = AttitudeEstimation()

    rospy.spin()

if __name__ == '__main__':
    main()
