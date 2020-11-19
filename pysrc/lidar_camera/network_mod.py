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

        # self.initializeWeights()

    def initializeWeights(self):
        for m in self.depth_cnn.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        for m in self.fc.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def getParamValueList(self):
        list_colorcnn_param_value = []
        list_depthcnn_param_value = []
        list_fc_param_value = []
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "color_cnn" in param_name:
                # print("color_cnn: ", param_name)
                list_colorcnn_param_value.append(param_value)
            if "depth_cnn" in param_name:
                # print("depth_cnn: ", param_name)
                list_depthcnn_param_value.append(param_value)
            if "fc" in param_name:
                # print("fc: ", param_name)
                list_fc_param_value.append(param_value)
        # print("list_colorcnn_param_value: ",list_colorcnn_param_value)
        # print("list_depthcnn_param_value: ",list_depthcnn_param_value)
        # print("list_fc_param_value: ",list_fc_param_value)
        return list_colorcnn_param_value, list_depthcnn_param_value, list_fc_param_value

    def forward(self, inputs_color, inputs_depth):
        ## cnn
        features_color = self.color_cnn(inputs_color)
        features_depth = self.depth_cnn(inputs_depth)
        # print("out_color_cnn: ", features_color.size())
        # print("out_depth_cnn: ", features_depth.size())
        ## concat
        features_color = torch.flatten(features_color, 1)
        features_depth = torch.flatten(features_depth, 1)
        features = torch.cat((features_color, features_depth), dim=1)
        ## fc
        outputs = self.fc(features)
        l2norm = torch.norm(outputs[:, :3].clone(), p=2, dim=1, keepdim=True)
        outputs[:, :3] = torch.div(outputs[:, :3].clone(), l2norm)  #L2Norm, |(gx, gy, gz)| = 1
        return outputs

##### test #####
# import data_transform_mod
# ## color image
# color_img_path = "../../../dataset_image_to_gravity/AirSim/example/camera_0.jpg"
# color_img_pil = Image.open(color_img_path)
# ## depth image
# depth_img_path = "../../../dataset_image_to_gravity/AirSim/example/lidar.npy"
# depth_img_numpy = np.load(depth_img_path)
# ## label
# acc_list = [0, 0, 1]
# acc_numpy = np.array(acc_list)
# ## trans param
# resize = 224
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# ## transform
# transform = data_transform_mod.DataTransform(resize, mean, std)
# color_img_trans, depth_img_trans, _ = transform(color_img_pil, depth_img_numpy, acc_numpy, phase="train")
# ## network
# net = Network(resize, dim_fc_out=3, use_pretrained_vgg=True)
# print(net)
# list_colorcnn_param_value, list_depthcnn_param_value, list_fc_param_value = net.getParamValueList()
# print("len(list_colorcnn_param_value) = ", len(list_colorcnn_param_value))
# print("len(list_depthcnn_param_value) = ", len(list_depthcnn_param_value))
# print("len(list_fc_param_value) = ", len(list_fc_param_value))
# ## prediction
# inputs_color = color_img_trans.unsqueeze_(0)
# inputs_depth = depth_img_trans.unsqueeze_(0)
# print("inputs_color.size() = ", inputs_color.size())
# print("inputs_depth.size() = ", inputs_depth.size())
# outputs = net(inputs_color, inputs_depth)
# print("outputs.size() = ", outputs.size())
# print("outputs = ", outputs)
