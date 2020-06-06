#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

class GravityPrediction:
    def __init__(self, net, size, mean, std):
        rospy.Subscriber("/image_raw", Image, self.callback)
        self.bridge = CvBridge()
        self.img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def callback(msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("cv_img.shape = ", cv_img.shape)
            tensor_img = torch.from_numpy(cv_img.transpose((2, 1, 0)))
            inputs = self.img_transform(cv_img)
        except CvBridgeError as e:
            print(e)

def main():
    rospy.init_node('gravity_prediction', anonymous=True)

    ## device check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = ", device)
    ## network
    net = models.vgg16()
    net.features[26] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
    net.features = nn.Sequential(*list(net.features.children())[:-3])
    net.classifier = nn.Sequential(
        nn.Linear(in_features=73728, out_features=18, bias=True),
        nn.ReLU(True),
        nn.Linear(in_features=18, out_features=3, bias=True)
    )
    print(net)
    ## size, mean, std
    size = 224  #VGG16
    mean = ([0.5, 0.5, 0.5])
    std = ([0.25, 0.25, 0.25])

    gravity_prediction = GravityPrediction(net, size, mean, std)

    rospy.spin()

if __name__ == '__main__':
    main()
