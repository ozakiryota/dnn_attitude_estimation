#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image

import torch
from torchvision import models
import torch.nn as nn

def callback(msg):
    # print("test")
    a = 0

def prediction():
    rospy.init_node('prediction', anonymous=True)
    rospy.Subscriber("/image_raw", Image, callback)

    ## device check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = ", device)
    ## network
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.features[26] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
    net.features = nn.Sequential(*list(net.features.children())[:-3])
    net.classifier = nn.Sequential(
        nn.Linear(in_features=73728, out_features=18, bias=True),
        nn.ReLU(True),
        nn.Linear(in_features=18, out_features=3, bias=True)
    )
    print(net)

    rospy.spin()

if __name__ == '__main__':
    prediction()
