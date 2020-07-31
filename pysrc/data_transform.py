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

##### test #####
# ## trans param
# size = 224  #VGG16
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# ## image
# image_file_path = "../dataset/example.jpg"
# img = Image.open(image_file_path)
# print("img.size = ", img.size)
# ## label
# g_list = [0, 0, 9.81]
# acc = np.array(g_list)
# ## transform
# transform = data_transform(size, mean, std)
# img_trans, acc_trans = transform(img, acc, phase="train")
# print("acc_trans", acc_trans)
# ## tensor -> numpy
# img_trans_numpy = img_trans.numpy().transpose((1, 2, 0))  #(rgb, h, w) -> (h, w, rgb)
# img_trans_numpy = np.clip(img_trans_numpy, 0, 1)
# print("img_trans_numpy.shape = ", img_trans_numpy.shape)
# ## save
# # save_path = "../dataset/augmented_example.jpg"
# # img_pil = Image.fromarray(np.uint8(255*img_trans_numpy))
# # img_pil.save(save_path)
# # print("saved: ", save_path)
# ## imshow
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(img_trans_numpy)
# plt.show()
