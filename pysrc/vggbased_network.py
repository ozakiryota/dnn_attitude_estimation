from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

import data_transform

class OriginalNet(nn.Module):
    def __init__(self):
        super(OriginalNet, self).__init__()

        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.fc = nn.Sequential(
            nn.Linear(25088, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(100, 18),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(18, 9)    #(x, y, z, sx, sy, sz, corr_xy, corr_yz, corr_zx)
        )
        self.copyVggParam(vgg)

    def copyVggParam(self, vgg):
        list_vgg_param_name = []
        for param_name, _ in vgg.named_parameters():
            list_vgg_param_name.append(param_name)
        for param_name, param_value in self.named_parameters():
            if param_name in list_vgg_param_name:
                # print("copy vgg: ", param_name)
                vgg.state_dict()[param_name].requires_grad = True
                self.state_dict()[param_name] = vgg.state_dict()[param_name]

    def getParamValueList(self):
        list_cnn_param_value = []
        list_fc_param_value = []
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "features" in param_name:
                # print("features: ", param_name)
                list_cnn_param_value.append(param_value)
            if "fc" in param_name:
                # print("fc: ", param_name)
                list_fc_param_value.append(param_value)
        # print("list_cnn_param_value: ",list_cnn_param_value)
        # print("list_fc_param_value: ",list_fc_param_value)
        return list_cnn_param_value, list_fc_param_value

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
        # x[:, 3:6] = torch.exp(x[:, 3:6])    #(sx, sy, sz) > 0
        # x[:, 6:9] = torch.tanh(x[:, 6:9])   #1 > (corr_xy, corr_yz, corr_zx) > -1
        # print("x[:, :3] = ", x[:, :3])
        # print("x[:, 3:6] = ", x[:, 3:6])
        # print("x[:, 6:9] = ", x[:, 6:9])
        return x

##### test #####
# ## network
# net = OriginalNet()
# print(net)
# list_cnn_param_value, list_fc_param_value = net.getParamValueList()
# # print(list_fc_param_value)
# ## image
# image_file_path = "../dataset/example.jpg"
# img = Image.open(image_file_path)
# ## label
# g_list = [0, 0, 9.81]
# acc = np.array(g_list)
# ## trans param
# size = 224  #VGG16
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# ## transform
# transform = data_transform.data_transform(size, mean, std)
# img_trans, _ = transform(img, acc, phase="train")
# ## prediction
# inputs = img_trans.unsqueeze_(0)
# print("inputs.size() = ", inputs.size())
# outputs = net(inputs)
# print("outputs.size() = ", outputs.size())
