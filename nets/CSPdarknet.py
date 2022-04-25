#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 10:36
# @Author  : shiman
# @File    : CSPdarknet.py
# @describe:

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """Mish激活函数, 公式Mish = x * tanh(ln(1+e^x))"""
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        out = x * torch.tanh(F.softplus(x))
        return out


class BasicConv(nn.Module):
    """基础卷积块：卷积+标准化+激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResBlock(nn.Module):
    """内部堆叠、基础残差块"""
    def __init__(self, channels, hidden_channels=None):
        super(ResBlock, self).__init__()

        if hidden_channels == None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        out = x+self.block(x)
        return out


class ResBlock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(ResBlock_body, self).__init__()
        # stride=2 宽高压缩
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            # 建立一个大的残差边
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            # 主干部分对num_blocks进行循环，循环内部是残差结构
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                ResBlock(out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )

            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:
            # 建立一个大的残差边
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)
            # 主干部分对num_blocks进行循环，循环内部是残差结构
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[ResBlock(out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )

            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        # 将残差便堆叠回来
        x = torch.cat([x1,x0], dim=1)
        # 最后对通道数进行整合
        x = self.concat_conv(x)
        return x


class CSPDarkNet(nn.Module):
    def __init__(self, layers):
        super(CSPDarkNet, self).__init__()

        self.inplanes = 32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            ResBlock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            ResBlock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            ResBlock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            ResBlock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            ResBlock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False),
        ])

        self.num_features = 1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)

        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return out3, out4, out5

def darknet53(pretrained):
    model = CSPDarkNet([1,2,8,8,4])
    if pretrained:
        model.load_state_dict(torch.load('model_data/CSPdarknet53_backbone_weights.pth'))
    return model



