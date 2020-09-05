#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import QuaternionStamped

from cv_bridge import CvBridge, CvBridgeError

import cv2
from PIL import Image
import math

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

import network_regression

class AttitudeEstimation:
    def __init__(self, frame_id, device, size, mean, std, net):
        ## subscriber
        self.sub_imgae = rospy.Subscriber("/image_raw", ImageMsg, self.callbackImage)
        ## publisher
        self.pub_vector = rospy.Publisher("/dnn/g_vector", Vector3Stamped, queue_size=1)
        ## msg
        self.v_msg = Vector3Stamped()
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
        ## tensor to numpy
        outputs = outputs[0].cpu().detach().numpy()
        ## Vector3Stamped
        self.v_msg.vector.x = -outputs[0]
        self.v_msg.vector.y = -outputs[1]
        self.v_msg.vector.z = -outputs[2]

    def publication(self, stamp):
        ## Vector3Stamped
        self.v_msg.header.stamp = stamp
        self.v_msg.header.frame_id = self.frame_id
        self.pub_vector.publish(self.v_msg)

def main():
    ## Node
    rospy.init_node('attitude_estimation', anonymous=True)
    ## Param
    weights_path = rospy.get_param("/weights_path", "../weights/regression.pth")
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
    net = network_regression.OriginalNet()
    print(net)
    net.to(device)
    ## Load weights
    weights_was_saved_in_same_device = True
    if weights_was_saved_in_same_device:
        ## saved in CPU -> load in CPU, saved in GPU -> load in GPU
        print("Loaded: GPU -> GPU or CPU -> CPU")
        load_weights = torch.load(weights_path)
    else:
        ## saved in GPU -> load in CPU
        print("Loaded: GPU -> CPU")
        load_weights = torch.load(weights_path, map_location={"cuda:0": "cpu"})
    net.load_state_dict(load_weights)
    ## set as eval
    net.eval()

    attitude_estimation = AttitudeEstimation(frame_id, device, size, mean, std, net)

    rospy.spin()

if __name__ == '__main__':
    main()
