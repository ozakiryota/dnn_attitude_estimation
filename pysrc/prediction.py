#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import QuaternionStamped
from tf.transformations import quaternion_from_euler

from cv_bridge import CvBridge, CvBridgeError

import cv2
from PIL import Image
import math

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

class AttitudeEstimation:
    def __init__(self, device, size, mean, std, net):
        self.sub = rospy.Subscriber("/image_raw", ImageMsg, self.callback)
        self.pub = rospy.Publisher("/dnn_attitude", QuaternionStamped, queue_size=1)
        self.bridge = CvBridge()
        self.device = device
        self.img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.net = net

    def callback(self, msg):
        print("----------")
        start_clock = rospy.get_time()
        try:
            img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("img_cv.shape = ", img_cv.shape)
            acc = self.dnn_prediction(img_cv)
            q_msg = self.acc_to_attitude(acc)
            q_msg.header.stamp = msg.header.stamp
            self.publication(q_msg)
        except CvBridgeError as e:
            print(e)
        print("Period [s]: ", rospy.get_time() - start_clock, "Frequency [hz]: ", 1/(rospy.get_time() - start_clock))

    def dnn_prediction(self, img_cv):
            img_pil = self.cv_to_pil(img_cv)
            img_transformed = self.img_transform(img_pil)
            inputs = img_transformed.unsqueeze_(0)
            inputs = inputs.to(self.device)
            outputs = self.net(inputs)
            print("outputs = ", outputs)
            return outputs

    def cv_to_pil(self, img_cv):
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv)
        return img_pil

    def acc_to_attitude(self, acc):
        acc = acc[0].detach().numpy()
        print(acc)
        r = math.atan2(acc[1], acc[2])
        p = math.atan2(-acc[0], acc[1] * math.sin(r) + acc[2] * math.cos(r))
        y = 0
        print("r = ", r, ", p = ", p)
        q_tf = quaternion_from_euler(r, p, y)
        q_msg = QuaternionStamped()
        q_msg.quaternion.x = q_tf[0]
        q_msg.quaternion.y = q_tf[1]
        q_msg.quaternion.z = q_tf[2]
        q_msg.quaternion.w = q_tf[3]
        return q_msg

    def publication(self, q_msg):
        q_msg.header.frame_id = "/base_link"
        self.pub.publish(q_msg)

def main():
    ## Node
    rospy.init_node('attitude_estimation', anonymous=True)
    ## Param
    weights_path = rospy.get_param("/weights_path", "weights.pth")
    print("weights_path = ", weights_path)

    ## Device check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = ", device)
    # print("torch.cuda.current_device() = ", torch.cuda.current_device())
    ## size, mean, std
    size = 224  #VGG16
    mean = ([0.5, 0.5, 0.5])
    std = ([0.25, 0.25, 0.25])
    ## Network
    net = models.vgg16()
    net.features[26] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
    net.features = nn.Sequential(*list(net.features.children())[:-3])
    net.avgpool = nn.Sequential()
    net.classifier = nn.Sequential(
        nn.Linear(in_features=73728, out_features=18, bias=True),
        nn.ReLU(True),
        nn.Linear(in_features=18, out_features=3, bias=True)
    )
    print(net)
    net.to(device)
    ## Load weights
    was_saved_in_same_device = False
    if was_saved_in_same_device:
        ## saved in CPU -> load in CPU, saved in GPU -> load in GPU
        load_weights = torch.load(weights_path)
    else:
        ## saved in GPU -> load in CPU
        load_weights = torch.load(weights_path, map_location={"cuda:0": "cpu"})
    net.load_state_dict(load_weights)
    ## set as eval
    net.eval()

    attitude_estimation = AttitudeEstimation(device, size, mean, std, net)

    rospy.spin()

if __name__ == '__main__':
    main()
