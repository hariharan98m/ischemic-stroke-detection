from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
import torch.nn.functional as F

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN8s(nn.Module):

    def __init__(self, name, pretrained_net, n_class=2):
        super().__init__()

        self.name = name
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu  = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim = -1)
        self.sigmoid = nn.Sigmoid()
        self.bn4096 = nn.BatchNorm2d(4096, affine=False)
        self.bn1024 = nn.BatchNorm2d(1024, affine=False)
        self.bn512 = nn.BatchNorm2d(512, affine=False)
        self.bn256 = nn.BatchNorm2d(256, affine=False)
        self.times2Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # convert the output feature map of the VGG 512x7x7 from 512 to 4096.
        self.fcn_conv1 =  nn.Conv2d(512, 4096, kernel_size=3, padding=1)
        # self.fcn_conv2 = nn.Conv2d(4096, 4096, kernel_size=3, padding=1)

        self.upsample_times4_conv1 = nn.Conv2d(4096, 1024, kernel_size=3, stride=1, padding=1)
        self.upsample_times4_conv2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)

        self.upsample_times2_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)


        # last in_channels = 256 + 512 + 4096
        in_channels = 256 + 512 + 512

        # modification starts here.
        self.fcn_conv_from_stacked1 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, padding=0)

        self.fcn_conv_from_stacked2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        self.last = nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0)

        self.cls = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def print_layer_shape(self, name, layer):
        print(name, layer.size().numpy())

    def forward(self, x):
        output = self.pretrained_net(x)
        pool5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        pool4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        pool3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        # 7x7x512 to 7x7x4096
        prop = self.leaky_relu(self.fcn_conv1(pool5))
        prop = self.bn4096(prop)

        # 2nd level conv block. 7x7x4096 to 7x7x4096
        # prop = self.relu(self.fcn_conv2(prop))
        # prop = self.bn4096(prop)

        # upsample 7x7x4096 by 4 times. gives 28x28x4096
        score_5 = self.times2Upsample(prop)
        score_5 = self.bn1024(self.leaky_relu(self.upsample_times4_conv1(score_5)))
        score_5 = self.times2Upsample(score_5)
        score_5 = self.bn512(self.leaky_relu(self.upsample_times4_conv2(score_5)))

        # upsample the pool4 (14x14x512) by 2 times to give 28x28x512
        score_4 = self.times2Upsample(pool4)
        score_4 = self.bn512(self.leaky_relu(self.upsample_times2_conv(score_4)))

        # take pool3 (28 x 28 x 256) as it is.
        score_3 = pool3

        # torch cat tensor
        score = torch.cat((score_5, score_4, score_3), dim = 1)

        # new code: make in_channels to n_class channels

        # 256 + 512 + 512 to 512 channels
        in_channels_classes = self.leaky_relu(self.fcn_conv_from_stacked1(score))
        in_channels_classes = self.bn512(in_channels_classes)
        # 512 to 256 channels
        in_channels_classes = self.leaky_relu(self.fcn_conv_from_stacked2(in_channels_classes))
        in_channels_classes = self.bn256(in_channels_classes)
        # 256 to n_class channels
        out_classes = self.leaky_relu(self.last(in_channels_classes))

        # final classification conv layer
        cls_layer = F.log_softmax(self.cls(out_classes), dim=1)

        return cls_layer


    def _initialize_weights(self):
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     m.weight.data.zero_()
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
