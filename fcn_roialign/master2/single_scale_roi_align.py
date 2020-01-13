from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
import torch.nn.functional as F
from torchvision.ops import RoIAlign
import pdb

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


class Unet(nn.Module):

    def __init__(self, name, pretrained_net, n_class=2):
        super().__init__()

        self.name  = name
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu  = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim = -1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p = 0.2)
        self.bn4096 = nn.BatchNorm2d(4096, affine=True)
        self.bn1024 = nn.BatchNorm2d(1024, affine=True)
        self.bn512 = nn.BatchNorm2d(512, affine=True)
        self.bn256 = nn.BatchNorm2d(256, affine=True)
        self.bn128 = nn.BatchNorm2d(128, affine=True)
        self.bn64 = nn.BatchNorm2d(64)
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
        self.fcn_conv_from_stacked1 = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)
        self.fcn_conv_from_stacked2_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.fcn_conv_from_stacked2_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.fcn_conv_from_stacked3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.fcn_conv_from_stacked4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.fcn_conv_from_stacked5_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fcn_conv_from_stacked5_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.last = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)

        self.cls = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0)

        # extracting RoI of size 56x56 on 224x224
        self.scale_1_roi_align = RoIAlign((28, 28), spatial_scale=0.5,
                                          sampling_ratio=2)  # wil look as 28x28 on the 112x112 image
        self.scale_2_roi_align = RoIAlign((14, 14), spatial_scale=0.25,
                                          sampling_ratio=2)  # will look at 14x14 on the 56x56 image
        self.scale_3_roi_align = RoIAlign((7, 7), spatial_scale=0.125,
                                          sampling_ratio=2)  # will look at 7x7 on the 28x28 image

        self._initialize_weights()

    def print_layer_shape(self, name, layer):
        print(name, layer.size().numpy())

    def forward(self, inputs):
        (x, cut_boxes) = inputs

        output = self.pretrained_net(x)

        pool28_concat = output['concat_28'] # size=(N, 256, 28, 28)
        pool28 = output['scale_28'] # size=(N, 256, 28, 28)
        pool56 = output['scale_56'] # size=(N, 128, 56, 56)
        pool112 = output['scale_112'] # size=(N, 64, 112, 112)


        pdb.set_trace()
        scale1_28, scale2_14, scale3_7, scale4_7 = self.scale_1_roi_align(pool112, cut_boxes), \
                                       self.scale_2_roi_align(pool56, cut_boxes), \
                                       self.scale_3_roi_align(pool28, cut_boxes), \
                                       self.scale_3_roi_align(pool28_concat, cut_boxes)


        print(scale1_28, scale2_14, scale3_7, scale4_7)

        # 7x7x512 to 7x7x4096
        prop = self.relu(self.fcn_conv1(scale4_7))
        prop = self.bn4096(prop)

        # 2nd level conv block. 7x7x4096 to 7x7x4096
        # prop = self.relu(self.fcn_conv2(prop))
        # prop = self.bn4096(prop)

        # upsample 7x7x4096 by 4 times. gives 28x28x4096
        score_5 = self.times2Upsample(prop)
        score_5 = self.bn1024(self.relu(self.upsample_times4_conv1(score_5)))
        score_5 = self.times2Upsample(score_5)
        score_5 = self.bn512(self.relu(self.upsample_times4_conv2(score_5)))

        # upsample the pool4 (14x14x512) by 2 times to give 28x28x512
        score_4 = self.times2Upsample(scale2_14)
        score_4 = self.bn512(self.relu(self.upsample_times2_conv(score_4)))

        # take pool3 (28 x 28 x 256) as it is.
        score_3 = scale1_28

        # torch cat tensor
        score = torch.cat((score_5, score_4, score_3), dim = 1)

        # new code: make in_channels to n_class channels

        # 256 + 512 + 512 to 512 channels
        x = self.relu(self.fcn_conv_from_stacked1(score))
        x = self.bn512(x)  # 28x28x512

        # still in 512 channels
        # 512 to 256 channels
        parallel_conv1 = self.leaky_relu(self.fcn_conv_from_stacked2_1(x))
        parallel_conv2 = self.leaky_relu(self.fcn_conv_from_stacked2_2(x))
        # get res from prev block and add to current ptr
        res = torch.add(torch.add(parallel_conv1, parallel_conv2), self.dropout(pool3))
        res = self.bn256(res)
        # now have 28 x 28 x 256

        # go from 28 to 56 spatial
        x = self.times2Upsample(res)   # 56 spatial
        # 256 to 128 channels
        x = self.leaky_relu(self.fcn_conv_from_stacked3(x))
        x = self.bn128(x)   # 56x56x128
        res = torch.add(x, self.dropout(pool2))

        # go from 56 to 112 spatial
        x = self.times2Upsample(res)  # 112 spatial
        # 128 to 64 channels
        x = self.leaky_relu(self.fcn_conv_from_stacked4(x))
        x = self.bn64(x)  # 112x112x64
        res = torch.add(x, self.dropout(pool1))

        parallel1 = self.times2Upsample(res)  # back to 224
        parallel2 = self.leaky_relu(self.fcn_conv_from_stacked5_1(self.times2Upsample(res)))  # back to 224
        parallel3 = self.leaky_relu(self.fcn_conv_from_stacked5_2(self.times2Upsample(res)))  # back to 224
        res = self.bn64(parallel1+parallel2 + parallel3)

        out_classes = self.leaky_relu(self.last(res))    # go from 64 to 2

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
