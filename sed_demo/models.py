#!/usr/bin python
# -*- coding:utf-8 -*-


"""
This module contains code to define the used PyTorch model(s). It has been
borrowed from:

https://github.com/qiuqiangkong/audioset_tagging_cnn
https://github.com/yinkalario/General-Purpose-Sound-Recognition-Demo
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


# ##############################################################################
# # HELPERS
# ##############################################################################
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        """
        """
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


# ##############################################################################
# # CNN 9
# ##############################################################################
class Cnn9_GMP_64x64(nn.Module):
    """
    """

    EXPECTED_NUM_CLASSES = 527  # Model was trained with AudioSet classes

    def __init__(self, classes_num, strong_target_training=False):
        """
        """
        super().__init__()
        assert classes_num == self.EXPECTED_NUM_CLASSES, \
            f"Expected 527 AudioSet classes and got {classes_num}!"
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        self.init_weights()

    def init_weights(self):
        """
        """
        init_layer(self.fc_audioset)

    def get_bottleneck(self, x):
        """
        """
        x = x[:, None, :, :]  # (batch, 1, time, freqbins)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = self.conv_block4(x, pool_size=(1, 1), pool_type="avg")
        return x

    def forward(self, x):
        """
        Input: (batch_size, times_steps, freq_bins)
        """
        x = self.get_bottleneck(x)
        x = torch.mean(x, dim=3)
        (x, _) = torch.max(x, dim=2)
        output = torch.sigmoid(self.fc_audioset(x))
        return output
