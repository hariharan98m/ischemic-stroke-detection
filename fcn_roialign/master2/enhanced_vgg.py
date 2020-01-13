from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import numpy as np
import torch.nn.functional as F

from torchvision.ops import RoIAlign
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImprovedMaxPool(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ImprovedMaxPool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1 , bias=True)   # (224 + 2*1 - 5 )/2
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        conv_out1 = self.leaky_relu(self.conv1(x))
        maxpool_out2 = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        return self.dropout(torch.add(conv_out1, maxpool_out2))

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 4 # common is 3.
    for v in cfg:
        if v == 'M':
            layers += [ ImprovedMaxPool(in_channels, in_channels) ]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class EnhancedVGGNet(VGG):
    def __init__(self, freeze_max = True, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model], batch_norm=True))
        self.ranges = ranges['enhanced_'+model]

        # if pretrained:
        #     exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if freeze_max:
            for name, param in self.named_parameters():
                if name.startswith('features'):
                    num = int(name.lstrip('features.').split('.')[0])
                    if num ==0 and num>=21:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def _initialize_weights(self):

        # vgg16_params = models.vgg16(pretrained=True).state_dict()

        # just change the first filter size
        # vgg16_params['features.0.weight'] = torch.randn( 64, 4, 3, 3, requires_grad = True)

        # self.load_state_dict(vgg16_params)

        pass

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'enhanced_vgg16': ((0,7), (7, 14), (14, 24), (24,34), (34,44)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class StartBlock(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, same_c = False):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_aux_layer2_1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_aux_layer2_2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1, bias=True)

        self.conv_block3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, input):
        # spatial
        batch_size, c, h, w = list(input.size())

        layer1_out = self.leaky_relu(self.conv1(input))
        layer2_out = self.leaky_relu(self.conv2(layer1_out))
        block_out1 = RoIAlign((h // 2, w // 2), spatial_scale=1.0, sampling_ratio=2)(
            layer2_out,
            get_rois(batch_size, [0, 0, h-1, w -1]).to(device)
        )

        # aux connection from layer 2
        aux_from_layer2 =  self.leaky_relu(self.conv_aux_layer2_1(layer2_out))
        aux_layer2 =  self.leaky_relu(self.conv_aux_layer2_2(aux_from_layer2))
        block_out2 = self.dropout(aux_layer2)

        aux_from_layer1= torch.add(layer1_out, self.dropout(aux_from_layer2))
        block_out3 = self.leaky_relu(self.conv_block3(aux_from_layer1))

        return {
            'out1': torch.add(torch.add(block_out1, block_out2), block_out3),
            'out2': block_out2,
            'out3': block_out3
        }


class IntermediateBlock1(nn.Module):

    def __init__(self, in_channels, height, width):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)

        self.in_channels = in_channels

        self.conv_layer1 = nn.Conv2d( in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_layer2 = nn.Conv2d( 2 * in_channels, 2* in_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_block2 = nn.Conv2d(4 * in_channels, 2 * in_channels, kernel_size=3, stride=2, padding=1, bias=True)

        self.conv_prev_block = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.linear_units = []

        for i in range(height * width):
            # create a linear unit
            linear_unit = nn.Linear(in_channels, 1, bias = True)
            # add the units to a list
            self.linear_units.append(linear_unit.to(device))

            # register the weight and biases as learnable parameters
            self.register_parameter('linear_weight_' + str(i), linear_unit.weight)
            # for bias
            self.register_parameter('linear_biases_' + str(i), linear_unit.bias)


    def forward(self, out):
        # outs
        out1, out2, out3 = out['out1'], out['out2'], out['out3']

        # spatial
        batch_size, c, h, w = list(out1.size())

        # forward prop
        layer1_out = self.leaky_relu(self.conv_layer1(out1))
        layer2_out = self.leaky_relu(self.conv_layer2(layer1_out))
        block_out1 = RoIAlign((h // 2, w // 2), spatial_scale=1.0, sampling_ratio=2)(
            layer2_out,
            get_rois(batch_size, [0, 0, h-1, w -1]).to(device)
        )

        intermediate_curr_block = torch.cat([layer1_out, layer2_out], dim=1)
        block_out2 = self.leaky_relu(self.conv_block2(intermediate_curr_block))
        block_out2 = self.dropout(block_out2)

        prev_block = torch.cat([out2, out3], dim=1)
        prev_block_concat = self.leaky_relu(self.conv_prev_block(prev_block))

        out1_moveaxis = out1.permute(0, 2, 3, 1).view(batch_size, -1, c)

        # dense layers - time distributed
        attn_weights = []
        for item in range(out1_moveaxis.size(1)):
            attn_weights.append(self.linear_units[item](out1_moveaxis[:, item, :]))

        formatted = torch.stack(attn_weights).permute(1, 2, 0) # h*w elements of shape (batch_size , 1)
        softmax_formatted = nn.Softmax(dim=-1)(formatted)

        softmax_attn_matrix = softmax_formatted.view(batch_size, 1, h, w)

        # get the attention tensor for Out1 from the previous block.
        attn_out = torch.mul(prev_block_concat, softmax_attn_matrix)
        block_out3 = RoIAlign((h // 2, w // 2), spatial_scale=1.0, sampling_ratio=2)(
            attn_out,
            get_rois(batch_size, [0, 0, h-1, w -1]).to(device)
        )

        return {
            'out1': torch.add(torch.add(block_out1, block_out2), block_out3),
            'out2': block_out2,
            'out3': block_out3
        }



class IntermediateBlock2(nn.Module):

    def __init__(self, in_channels, height, width, same_c = False):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)
        self.same_c = same_c
        self.in_channels = in_channels

        if self.same_c:
            out_channels = in_channels
        else:
            out_channels = 2* in_channels

        self.conv_layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_layer3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_aux_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

        self.conv_block2 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

        self.conv_prev_block = nn.Conv2d(2 * in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.linear_units = []

        for i in range(height * width):
            # create a linear unit
            linear_unit = nn.Linear(in_channels, 1, bias=True)

            # add linear unit to pool
            self.linear_units.append(linear_unit)

            # register param, the bias first
            self.register_parameter('linear_bias_'+ str(i), linear_unit.bias)

            # register the weight param
            self.register_parameter('linear_weight_' + str(i), linear_unit.weight)


    def forward(self, out):
        # outs
        out1, out2, out3 = out['out1'], out['out2'], out['out3']

        # spatial
        batch_size, c, h, w = list(out1.size())

        # forward prop
        layer1_out = self.leaky_relu(self.conv_layer1(out1))
        layer2_out = self.leaky_relu(self.conv_layer2(layer1_out))
        layer3_out = self.leaky_relu(self.conv_layer3(layer2_out))
        roi_align_out = RoIAlign((h // 2, w // 2), spatial_scale=1.0, sampling_ratio=2)(
            layer2_out,
            get_rois(batch_size, [0, 0, h-1, w -1]).to(device)
        )
        aux_out = self.leaky_relu(self.conv_aux_out(self.dropout(layer3_out)))
        block_out1 = torch.add(roi_align_out, aux_out)

        intermediate_curr_block = torch.cat([layer1_out, layer2_out], dim=1)
        block_out2 = self.leaky_relu(
            self.conv_block2(intermediate_curr_block)
        )
        block_out2 = self.dropout(block_out2)

        prev_block = torch.cat([out2, out3], dim=1)
        prev_block_concat = self.leaky_relu(
            self.conv_prev_block(prev_block)
        )
        out1_moveaxis = out1.permute(0, 2, 3, 1).view(batch_size, -1, c)

        # dense layers - time distributed
        attn_weights = []
        for item in range(out1_moveaxis.size(1)):
            attn_weights.append(self.linear_units[item](out1_moveaxis[:, item, :]))

        formatted = torch.stack(attn_weights).permute(1, 2, 0) # h*w elements of shape (batch_size , 1)
        softmax_formatted = nn.Softmax(dim=-1)(formatted)

        softmax_attn_matrix = softmax_formatted.view(batch_size, 1, h, w)

        # get the attention tensor for Out1 from the previous block.
        attn_out = torch.mul(prev_block_concat, softmax_attn_matrix)
        block_out3 = RoIAlign((h // 2, w // 2), spatial_scale=1.0, sampling_ratio=2)(
            attn_out,
            get_rois(batch_size, [0, 0, h-1, w -1]).to(device)
        )

        return {
            'out1': torch.add(torch.add(block_out1, block_out2), block_out3),
            'out2': block_out2,
            'out3': block_out3
        }


class EnhancedBaseModel(nn.Module):
    def __init__(self, freeze_max = False):
        super().__init__()
        self.block1 = StartBlock(in_channels=4, out_channels=64, height=224, width=224)    # output is of shape (112, 112, 64) for input of shape (224, 224, 4)
        self.block2 = IntermediateBlock1(64, height=112, width=112)
        self.block3 = IntermediateBlock1(128, height=56, width=56)
        self.block4 = IntermediateBlock2(256, height=28, width= 28)
        self.block5 = IntermediateBlock2(512, height=14, width=14, same_c = True)

        self.penultimate_conv_1 = nn.Conv2d(512, 128, kernel_size=5, stride=1, padding=0, bias=True)
        self.penultimate_conv_2 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=0, bias=True)
        self.last_conv = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=True)

        if freeze_max:
            for param in super().parameters():
                param.requires_grad = False


    def forward(self, input):
        pool1 = self.block1(input)
        pool2 = self.block2(pool1)
        pool3 = self.block3(pool2)
        pool4 = self.block4(pool3)
        pool5 = self.block5(pool4)

        output_tensor = pool5['out1']

        # got output shape of 7x7
        block_outs = self.penultimate_conv_1(output_tensor)  # shape of (batch_size, 2, 1, 1)
        block_outs = self.penultimate_conv_2(block_outs)
        cls_layer = self.last_conv(block_outs)

        return {
            'x1': pool1['out1'],
            'x2': pool2['out1'],
            'x3': pool3['out1'],
            'x4': pool4['out1'],
            'x5': pool5['out1'],
            'classifier': F.log_softmax(cls_layer, dim=1)
        }


def get_rois(batch_size, coords):
    rois_np = np.zeros((batch_size, 5))
    for i in range(len(rois_np)):
        rois_np[i] = [i] + coords
    rois = rois_np.astype(np.float32)
    rois = torch.from_numpy(rois)
    return rois


