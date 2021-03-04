import numpy as np

import torch
from torchvision import models
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, num_images, resize=224, dim_fc_out=3, use_pretrained=True):
        super(Network, self).__init__()

        vgg = models.vgg16(pretrained=use_pretrained)
        self.features = vgg.features

        dim_fc_in = (resize//32)*(num_images*resize//32)*512
        self.fc = nn.Sequential(
            nn.Linear(dim_fc_in, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(100, 18),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(18, dim_fc_out)    #(x, y, z, L00, L10, L11, L20, L21, L22)
        )

    def forward(self, x):
        # print("cnn-in", x.size())
        x = self.features(x)
        # print("cnn-out", x.size())
        x = torch.flatten(x, 1)
        # print("fc-in", x.size())
        x = self.fc(x)
        # print("fc-out", x.size())
        l2norm = torch.norm(x[:, :3].clone(), p=2, dim=1, keepdim=True)
        x[:, :3] = torch.div(x[:, :3].clone(), l2norm)  #L2Norm, |(gx, gy, gz)| = 1
        return x
