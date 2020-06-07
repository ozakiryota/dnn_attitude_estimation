#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image as ImageMsg

from cv_bridge import CvBridge, CvBridgeError

import cv2
from PIL import Image

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

class GravityPrediction:
    def __init__(self, size, mean, std, net):
        self.sub = rospy.Subscriber("/image_raw", ImageMsg, self.callback)
        self.bridge = CvBridge()
        self.net = net
        self.img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def callback(self, msg):
        try:
            img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("img_cv.shape = ", img_cv.shape)
            img_pil = self.cv_to_pil(img_cv)
            img_transformed = self.img_transform(img_pil)
            inputs = img_transformed.unsqueeze_(0)
            outputs = self.net(inputs)
            print("outputs = ", outputs)
        except CvBridgeError as e:
            print(e)

    def cv_to_pil(self, img_cv):
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv)
        return img_pil

def main():
    ## Node
    rospy.init_node('gravity_prediction', anonymous=True)
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
    net.classifier = nn.Sequential(
        nn.Linear(in_features=73728, out_features=18, bias=True),
        nn.ReLU(True),
        nn.Linear(in_features=18, out_features=3, bias=True)
    )
    print(net)
    ## Load weights
    was_saved_in_same_device = False
    if was_saved_in_same_device:
        ## saved in CPU -> load in CPU, saved in GPU -> load in GPU
        load_weights = torch.load(weights_path)
    else:
        ## saved in GPU -> load in CPU
        load_weights = torch.load(weights_path, map_location={"cuda:0": "cpu"})
    net.load_state_dict(load_weights)

    gravity_prediction = GravityPrediction(size, mean, std, net)

    rospy.spin()

if __name__ == '__main__':
    main()
