#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Vector3Stamped

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
        print("--- regression_prediction ---")
        ## parameter-msg
        self.frame_id = rospy.get_param("/frame_id", "/base_link")
        print("self.frame_id = ", self.frame_id)
        self.num_cameras = rospy.get_param("/num_cameras", 1)
        print("self.num_cameras = ", self.num_cameras)
        ## parameter-dnn
        weights_path = rospy.get_param("/weights_path", "../../weights/mle.pth")
        print("weights_path = ", weights_path)
        resize = rospy.get_param("/resize", 224)
        print("resize = ", resize)
        mean_element = rospy.get_param("/mean_element", 0.5)
        print("mean_element = ", mean_element)
        std_element = rospy.get_param("/std_element", 0.5)
        print("std_element = ", std_element)
        ## subscriber
        self.list_sub = []
        for camera_idx in range(self.num_cameras):
            sub_image = rospy.Subscriber("/image_raw" + str(camera_idx), ImageMsg, self.callbackImage, callback_args=camera_idx, queue_size=1, buff_size=2**24)
            self.list_sub.append(sub_image)
        ## publisher
        self.pub_vector = rospy.Publisher("/dnn/g_vector", Vector3Stamped, queue_size=1)
        ## msg
        self.v_msg = Vector3Stamped()
        ## cv_bridge
        self.bridge = CvBridge()
        ## list
        self.list_img_cv = [np.empty(0)]*self.num_cameras
        self.list_got_new_img = [False]*self.num_cameras
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
        net = network_mod.Network(self.num_cameras, resize=resize, dim_fc_out=3, use_pretrained=False)
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

    def callbackImage(self, msg, camera_idx):
        print("----------")
        start_clock = rospy.get_time()
        try:
            img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("img_cv.shape = ", img_cv.shape)
            self.list_img_cv[camera_idx] = img_cv
            self.list_got_new_img[camera_idx] = True
            if all(self.list_got_new_img):
                outputs = self.dnnPrediction()
                self.inputToMsg(outputs)
                self.publication(msg.header.stamp)
                print("Period [s]: ", rospy.get_time() - start_clock, "Frequency [hz]: ", 1/(rospy.get_time() - start_clock))
        except CvBridgeError as e:
            print(e)

    def dnnPrediction(self):
        ## inference
        inputs = self.transformImage()
        print("inputs.size() = ", inputs.size())
        outputs = self.net(inputs)
        print("outputs = ", outputs)
        ## reset
        self.list_got_new_img = [False]*self.num_cameras
        print("self.list_got_new_img = ", self.list_got_new_img)
        return outputs

    def transformImage(self):
        for i in range(self.num_cameras):
            img_pil = self.cvToPIL(self.list_img_cv[i])
            img_tensor = self.img_transform(img_pil)
            if i == 0:
                combined_img_tensor = img_tensor
            else:
                # combined_img_tensor = torch.cat((combined_img_tensor, img_tensor), dim=2)
                combined_img_tensor = torch.cat((img_tensor, combined_img_tensor), dim=2)
        inputs = combined_img_tensor.unsqueeze_(0)
        inputs = inputs.to(self.device)
        return inputs

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
