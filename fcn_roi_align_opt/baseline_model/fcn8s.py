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

    def __init__(self, pretrained_net, n_class=2):
        super().__init__()

        self.n_class = n_class
        self.pretrained_net = pretrained_net

        self.relu  = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim = -1)

        self.fcn_conv1 =  nn.Conv2d(512, 4096, kernel_size=3, padding=1)
        self.fcn_bn1 = nn.BatchNorm2d(4096)

        self.fcn_conv2 = nn.Conv2d(4096, 4096, kernel_size=3, padding=1)
        self.fcn_bn2 = nn.BatchNorm2d(4096)

        self.deconv_fcn_conv1 = nn.ConvTranspose2d(4096, 4096, kernel_size=2, stride=5, padding=2, dilation=1, output_padding=0)
        self.deconv_fcn_bn1 = nn.BatchNorm2d(4096)

        self.deconv_fcn_conv2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0, dilation=1,
                                                   output_padding=0)
        self.deconv_fcn_bn2 = nn.BatchNorm2d(512)

        # last in_channels = 256 + 512 + 4096
        in_channels = 256 + 512 + 4096

        # modification starts here.
        self.fcn_conv_from_deconv1 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, padding=0)
        self.fcn_conv_from_deconv1_bn = nn.BatchNorm2d(512)

        self.fcn_conv_from_deconv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.fcn_conv_from_deconv2_bn = nn.BatchNorm2d(256)

        self.last = nn.Conv2d(256, n_class, kernel_size=1, stride=1, padding=0)
        self.last_bn = nn.BatchNorm2d(n_class)


        # do deconv 2 times up,
        self.deconv_last1 = nn.ConvTranspose2d(n_class, n_class, kernel_size=2, stride=2, padding=0, dilation=1,
                                              output_padding=0)
        # self.conv_last1 = nn.Conv2d(n_class, n_class, kernel_size=3, padding=1)
        self.deconv_last1_bn = nn.BatchNorm2d(n_class)

        # then do deconv 2 times up, to make it to 4.
        self.deconv_last2 = nn.ConvTranspose2d(n_class, n_class, kernel_size=2, stride=2, padding=0, dilation=1,
                                              output_padding=0)
        # self.conv_last2 = nn.Conv2d(n_class, n_class, kernel_size=3, padding=1)
        self.deconv_last2_bn = nn.BatchNorm2d(n_class)

        # then deconv 2 times up, further to make it to 8.
        self.deconv_last3 = nn.ConvTranspose2d(n_class, n_class, kernel_size=2, stride=2, padding=0, dilation=1,
                                              output_padding=0)
        # self.conv_last3 = nn.Conv2d(n_class, n_class, kernel_size=3, padding=1)
        self.deconv_last3_bn = nn.BatchNorm2d(n_class)

        self.cls = nn.Conv2d(n_class, n_class, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

        # modification ends here.



        # self.last = nn.Conv2d(in_channels, n_class, kernel_size=1, stride=1, padding=0)
        #
        # # do deconv 8 times up
        # self.deconv_last= nn.ConvTranspose2d(n_class, n_class, kernel_size=8, stride=8, padding=0, dilation=1,
        #                                            output_padding=0)
        #
        # self.deconv_last_bn = nn.BatchNorm2d(n_class)
        #
        # self.cls = nn.Conv2d(n_class, n_class, kernel_size=1, stride=1, padding=0)
        #
        # self._initialize_weights()

    def print_layer_shape(self, name, layer):
        print(name, layer.size().numpy())

    def forward(self, x):
        output = self.pretrained_net(x)
        pool5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        pool4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        pool3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)



        # 7x7x512 to 7x7x4096
        prop = self.relu(self.fcn_conv1(pool5))
        prop = self.fcn_bn1(prop)

        # 2nd level conv block. 7x7x4096 to 7x7x4096
        prop = self.relu(self.fcn_conv2(prop))
        prop = self.fcn_bn2(prop)

        # upsample 7x7x 4096 by 4 times. gives 28x28x4096
        score_5 = self.relu(self.deconv_fcn_conv1(prop))
        score_5 = self.deconv_fcn_bn1(score_5)

        # upsample the pool4 (14x14x512) by 2 times to give 28x28x512
        score_4 = self.relu(self.deconv_fcn_conv2(pool4))
        score_4 = self.deconv_fcn_bn2(score_4)

        # take pool3 (28 x 28 x 256) as it is.
        score_3 = pool3

        # torch cat tensor
        score = torch.cat((score_5, score_4, score_3), dim = 1)

        # new code: make in_channels to n_class channels

        # 256 + 512 + 4096 to 512 channels
        in_channels_classes = self.relu(self.fcn_conv_from_deconv1(score))
        in_channels_classes = self.fcn_conv_from_deconv1_bn(in_channels_classes)
        # 512 to 256 channels
        in_channels_classes = self.relu(self.fcn_conv_from_deconv2(in_channels_classes))
        in_channels_classes = self.fcn_conv_from_deconv2_bn(in_channels_classes)
        # 256 to n_class channels
        in_channels_classes = self.relu(self.last(in_channels_classes))
        deconv_to_match_spatial = self.last_bn(in_channels_classes)

        # do deconv twice.
        # x2 times
        deconv_to_match_spatial = self.relu(self.deconv_last1(deconv_to_match_spatial))
        # deconv_to_match_spatial = self.relu(self.conv_last1(deconv_to_match_spatial))
        deconv_to_match_spatial = self.deconv_last1_bn(deconv_to_match_spatial)
        # x2 times to make it to x4
        deconv_to_match_spatial = self.relu(self.deconv_last2(deconv_to_match_spatial))
        # deconv_to_match_spatial = self.relu(self.conv_last2(deconv_to_match_spatial))
        deconv_to_match_spatial = self.deconv_last2_bn(deconv_to_match_spatial)
        # x4 times to make it to x8
        deconv_to_match_spatial = self.relu(self.deconv_last3(deconv_to_match_spatial))
        # deconv_to_match_spatial = self.relu(self.conv_last3(deconv_to_match_spatial))
        score = self.deconv_last3_bn(deconv_to_match_spatial)

        # final classification conv layer
        cls_layer = self.relu(self.cls(score))

        log_softmax_activated_scores = F.log_softmax(cls_layer, dim=1)

        # end code

        # old code
        '''
                
        # final conv to n_class channels
        # score to classifier 28 x 28 x n_class
        score = self.last(score)
        
        # get deconved output at 8x upsampling to get 224 x 224 x n_class
        score = self.relu(self.deconv_last(score))
        score = self.deconv_last_bn(score)

        # final classification conv layer
        cls_layer = self.relu(self.cls(score))

        log_softmax_activated_scores = F.log_softmax(cls_layer, dim =1)
        
        '''

        return log_softmax_activated_scores




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
