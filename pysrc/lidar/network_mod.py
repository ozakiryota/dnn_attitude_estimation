from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, resize, dim_fc_out=3, use_pretrained_vgg=True):
        super(Network, self).__init__()

        vgg = models.vgg16(pretrained=use_pretrained_vgg)
        self.color_cnn = vgg.features

        conv_kernel_size = (3, 5)
        conv_padding = (1, 2)
        pool_kernel_size = (2, 4)
        self.depth_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=conv_kernel_size, stride=1, padding=conv_padding),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Conv2d(64, 128, kernel_size=conv_kernel_size, stride=1, padding=conv_padding),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Conv2d(128, 256, kernel_size=conv_kernel_size, stride=1, padding=conv_padding),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_kernel_size)
        )

        dim_fc_in = 512*(resize//32)*(resize//32) + 256*(32//pool_kernel_size[0]**3)*(1812//pool_kernel_size[1]**3)
        self.fc = nn.Sequential(
            nn.Linear(dim_fc_in, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(100, 18),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(18, dim_fc_out)
        )

    def forward(self, inputs_color, inputs_depth):
        ## cnn
        features_color = self.color_cnn(inputs_color)
        features_depth = self.depth_cnn(inputs_depth)
        ## concat
        features_color = torch.flatten(features_color, 1)
        features_depth = torch.flatten(features_depth, 1)
        features = torch.cat((features_color, features_depth), dim=1)
        ## fc
        outputs = self.fc(features)
        l2norm = torch.norm(outputs[:, :3].clone(), p=2, dim=1, keepdim=True)
        outputs[:, :3] = torch.div(outputs[:, :3].clone(), l2norm)  #L2Norm, |(gx, gy, gz)| = 1
        return outputs
