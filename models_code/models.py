import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)
    

def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


def random_mask(x, p, training):
    if training:
        return torch.bernoulli((1. - p) * torch.ones(x.shape)).cuda()
    else:
        return 1.


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
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
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Cnn9_GMP_64x64(nn.Module):
    def __init__(self, classes_num, strong_target_training=False):
        
        super(Cnn9_GMP_64x64, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc_audioset)

    def get_bottleneck(self, input):
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        
        return x

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = self.get_bottleneck(input)

        x = torch.mean(x, dim=3)
        (x, _) = torch.max(x, dim=2)
        
        output = torch.sigmoid(self.fc_audioset(x))
        
        return output


class Cnn13_GMP_64x64(nn.Module):
    def __init__(self, classes_num, strong_target_training=False):
        
        super(Cnn13_GMP_64x64, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_audioset)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        x *= random_mask(x, p=0.5, training=self.training)
        (x, _) = torch.max(x, dim=2)
        x = F.dropout(x, p=0.5, training=self.training)
        output = torch.sigmoid(self.fc_audioset(x))
        
        return output
        
        
class Cnn13small_GMP_64x64(nn.Module):
    def __init__(self, classes_num, strong_target_training=False):
        
        super(Cnn13small_GMP_64x64, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=256)
        self.conv_block5 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block6 = ConvBlock(in_channels=512, out_channels=512)

        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_audioset)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        x *= random_mask(x, p=0.5, training=self.training)
        (x, _) = torch.max(x, dim=2)
        x = F.dropout(x, p=0.5, training=self.training)
        output = torch.sigmoid(self.fc_audioset(x))
        
        return output



###
def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
    
    
def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, pool_size=(1, 1), downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.pool_size = pool_size

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.pool_size != (1, 1):
            out = F.avg_pool2d(out, kernel_size=self.pool_size)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.layer1 = self._make_layer(block, 64, layers[0], pool_size=(1, 1))
        self.layer2 = self._make_layer(block, 128, layers[1], pool_size=(2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], pool_size=(2, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], pool_size=(2, 2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, pool_size=(1, 1)):
        downsample = None

        if pool_size != (1, 1) or self.inplanes != planes * block.expansion:
            if pool_size == (1, 1):
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion),
                    nn.BatchNorm2d(planes * block.expansion),)
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion),
                    nn.BatchNorm2d(planes * block.expansion),
                    nn.AvgPool2d(kernel_size=pool_size),)

        layers = []
        layers.append(block(self.inplanes, planes, pool_size, downsample))
        # if expand: 
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
        
        
class ResNet50_GMP_64x64(nn.Module):
    def __init__(self, classes_num):
        super(ResNet50_GMP_64x64, self).__init__()
            
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.resnet = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], zero_init_residual=True)    
        self.post_conv = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=(1, 4), stride=(1, 1), padding=(0, 0), bias=False)
        self.post_bn = nn.BatchNorm2d(2048)
        self.audioset_fc = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weights()

    def init_weights(self):
        init_layer(self.post_conv)
        init_bn(self.post_bn)
        init_layer(self.audioset_fc)
            
    def forward(self, input): 
        x = input[:, None, :, :]
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.resnet(x)
        
        x = F.relu_(self.post_bn(self.post_conv(x)))
        x = x.squeeze()
        (x, _) = torch.max(x, dim=2)

        x = torch.sigmoid(self.audioset_fc(x))
        
        return x


###
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential()

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)

        return out


class DenseNet121_GMP_64x64(nn.Module):
    def __init__(self, classes_num):
        super(DenseNet121_GMP_64x64, self).__init__()
            
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.densenet = DenseNet(growth_rate=32, block_config=(5, 10, 20, 40), num_init_features=64)

        self.post_conv = nn.Conv2d(in_channels=1708, out_channels=2048, kernel_size=(1, 4), stride=(1, 1), padding=(0, 0), bias=False)
        self.post_bn = nn.BatchNorm2d(2048)

        self.audioset_fc = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.post_conv)
        init_bn(self.post_bn)
        init_layer(self.audioset_fc)
            
    def forward(self, input):
        x = input[:, None, :, :]
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.densenet(x)

        x = F.relu_(self.post_bn(self.post_conv(x)))
        x = x.squeeze()
        (x, _) = torch.max(x, dim=2)

        x = torch.sigmoid(self.audioset_fc(x))
        
        return x


# class DenseNet121b_GMP_64x64(nn.Module):
#     def __init__(self, classes_num):
#         super(DenseNet121b_GMP_64x64, self).__init__()
            
#         self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
#         self.densenet = DenseNetKqq(growth_rate=32, block_config=(6, 12, 24, 48, 32), num_init_features=64)

#         self.fc_final = nn.Linear(2048, classes_num, bias=True)

#         self.init_weights()

#     def init_weights(self):
#         init_layer(self.fc_final)
            
#     def forward(self, input):
#         x = input[:, None, :, :]
        
#         x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
#         x = self.densenet(x)

#         x = torch.mean(x, dim=3)
#         (x, _) = torch.max(x, dim=2)
#         x = torch.sigmoid(self.fc_final(x))
        
#         return x
