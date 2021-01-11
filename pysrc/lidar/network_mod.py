import numpy as np

import torch
from torchvision import models
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, dim_fc_out=3):
        super(Network, self).__init__()

        conv_kernel_size = (3, 5)
        conv_padding = (1, 2)
        pool_kernel_size = (2, 4)
        self.cnn = nn.Sequential(
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

        dim_fc_in = 256*(32//pool_kernel_size[0]**3)*(1812//pool_kernel_size[1]**3)
        self.fc = nn.Sequential(
            nn.Linear(dim_fc_in, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(100, 18),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(18, dim_fc_out)
        )

    def forward(self, x):
        # print("cnn-in", x.size())
        x = self.cnn(x)
        # print("cnn-out", x.size())
        x = torch.flatten(x, 1)
        # print("fc-in", x.size())
        x = self.fc(x)
        # print("fc-out", x.size())
        l2norm = torch.norm(x[:, :3].clone(), p=2, dim=1, keepdim=True)
        x[:, :3] = torch.div(x[:, :3].clone(), l2norm)  #L2Norm, |(gx, gy, gz)| = 1
        return x
