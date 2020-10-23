from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms

class data_transform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            "val": transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, acc, phase="train"):
        if phase == "train":
        # if (phase == "train") or (phase == "val"):
            ## random
            angle_deg = random.uniform(-10.0, 10.0)
            angle_rad = angle_deg / 180 * math.pi
            # print("angle_deg = ", angle_deg)

            ## vector rotation
            rot = np.array([
                [1, 0, 0],
                [0, math.cos(-angle_rad), -math.sin(-angle_rad)],
                [0, math.sin(-angle_rad), math.cos(-angle_rad)]
            ])
            acc = np.dot(rot, acc)

            ## image rotation
            img = img.rotate(angle_deg)

        img_tensor = self.data_transform[phase](img)
        acc = acc.astype(np.float32)
        acc = acc / np.linalg.norm(acc)
        acc_tensor = torch.from_numpy(acc)

        return img_tensor, acc_tensor
