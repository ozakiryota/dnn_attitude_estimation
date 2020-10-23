from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms

class DataTransform():
    def __init__(self, resize, mean, std, num_images=-1):
        self.num_images = num_images
        self.img_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img_path_list, acc_numpy, phase="train"):
        self.setNumImages(img_path_list)
        camera_angle = 0.0
        if phase == "train":
            img_path_list, camera_angle = self.randomize_img_order(img_path_list)
            acc_numpy = self.rotateVector(acc_numpy, camera_angle)
        combined_img_tensor = self.combineImages(img_path_list)
        acc_numpy = acc_numpy.astype(np.float32)
        acc_numpy = acc_numpy / np.linalg.norm(acc_numpy)
        acc_tensor = torch.from_numpy(acc_numpy)
        return combined_img_tensor, acc_tensor

    def setNumImages(self, img_path_list):
        if self.num_images < 0:
            self.num_images = len(img_path_list)

    def randomize_img_order(self, img_path_list):
        slide = random.randint(0, len(img_path_list)-1)
        rand_img_path_list = [img_path_list[-len(img_path_list)+i+slide] for i in range(len(img_path_list))]
        camera_angle = 2*math.pi/len(img_path_list)*slide
        # camera_angle = -camera_angle	#NED->NEU
        # print("slide = ", slide)
        # print("camera_angle/math.pi*180 = ", camera_angle/math.pi*180)
        return rand_img_path_list, camera_angle

    def combineImages(self, img_path_list):
        for i in range(self.num_images):
            img_tensor = self.img_transform(Image.open(img_path_list[i]))
            if i == 0:
                combined_img_tensor = img_tensor
            else:
                # combined_img_tensor = torch.cat((combined_img_tensor, img_tensor), dim=2)
                combined_img_tensor = torch.cat((img_tensor, combined_img_tensor), dim=2)
        return combined_img_tensor

    def rotateVector(self, acc_numpy, camera_angle):
        rot = np.array([
    	    [math.cos(-camera_angle), -math.sin(-camera_angle), 0.0],
    	    [math.sin(-camera_angle), math.cos(-camera_angle), 0.0],
    	    [0.0, 0.0, 1.0]
    	])
        rot_acc_numpy = np.dot(rot, acc_numpy)
        return rot_acc_numpy

    # def getConcatH(self, img_l, img_r):
    #     dst = Image.new('RGB', (img_l.width + img_r.width, img_l.height))
    #     dst.paste(img_l, (0, 0))
    #     dst.paste(img_r, (img_l.width, 0))
    #     return dst

##### test #####
# ## trans param
# resize = 224
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# ## image
# # img_path_list = [
# #     "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_0.jpg",
# #     "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_288.jpg",
# #     "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_216.jpg",
# #     "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_144.jpg",
# #     "../../../dataset_image_to_gravity/AirSim/5cam/example/camera_72.jpg"
# # ]
# img_path_list = [
#     "../../../dataset_image_to_gravity/AirSim/4cam/example/camera_0.jpg",
#     "../../../dataset_image_to_gravity/AirSim/4cam/example/camera_270.jpg",
#     "../../../dataset_image_to_gravity/AirSim/4cam/example/camera_180.jpg",
#     "../../../dataset_image_to_gravity/AirSim/4cam/example/camera_90.jpg"
# ]
# ## label
# acc_list = [1, 0, 0]
# acc_numpy = np.array(acc_list)
# ## transform
# transform = DataTransform(resize, mean, std)
# img_trans, acc_trans = transform(img_path_list, acc_numpy)
# print("acc_trans", acc_trans)
# ## tensor -> numpy
# img_trans_numpy = img_trans.numpy().transpose((1, 2, 0))  #(rgb, h, w) -> (h, w, rgb)
# img_trans_numpy = np.clip(img_trans_numpy, 0, 1)
# print("img_trans_numpy.shape = ", img_trans_numpy.shape)
# ## save
# save_path = "../../save/combine_example.jpg"
# img_pil = Image.fromarray(np.uint8(255*img_trans_numpy))
# img_pil.save(save_path)
# print("saved: ", save_path)
# ## imshow
# for i in range(len(img_path_list)):
#     plt.subplot2grid((2, len(img_path_list)), (0, len(img_path_list)-i-1))
#     plt.imshow(Image.open(img_path_list[i]))
# plt.subplot2grid((2, len(img_path_list)), (1, 0), colspan=len(img_path_list))
# plt.imshow(img_trans_numpy)
# plt.show()
