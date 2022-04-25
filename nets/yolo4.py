#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 15:18
# @Author  : shiman
# @File    : yolo4.py
# @describe:


from collections import OrderedDict

import torch
import torch.nn as nn

from .CSPdarknet import darknet53

def conv2d(in_filters, out_filters, kernel_size, stride=1):
    pad = (kernel_size - 1) //2 if kernel_size else 0

    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_filters)),
        ('relu', nn.LeakyReLU(0.1)),
    ]))

class Upsample(nn.Module):
    """卷积+上采用"""
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        return self.upsample(x)


def make_three_conv(filter_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filter_list[0], 1),
        conv2d(filter_list[0], filter_list[1], 3),
        conv2d(filter_list[1], filter_list[0], 1)
    )
    return m


def make_five_conv(filter_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filter_list[0], 1),
        conv2d(filter_list[0], filter_list[1], 3),
        conv2d(filter_list[1], filter_list[0], 1),
        conv2d(filter_list[0], filter_list[1], 3),
        conv2d(filter_list[1], filter_list[0], 1)
    )
    return m


def yolo_head(filters_list, in_fliters):
    m = nn.Sequential(
        conv2d(in_fliters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1)
    )
    return m


class SpatialPyramidPooling(nn.Module):
    """SSP结构，进行不同大小池化，池化后堆叠"""
    def __init__(self, pool_sizes=[5,9,13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList(
            [nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes]
        )

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features+[x], dim=1)

        return features


class YoloBody(nn.Module):
    def __init__(self, anchor_mask, num_classes, pretrained=False):
        super(YoloBody, self).__init__()
        # 获取CSPdarknet53的主干模型，获取三个有效特征层
        # (52,52,256), (26,26,512), (13,13,1024)
        self.backbone = darknet53(pretrained)
        #
        self.conv1 = make_three_conv([512,1024], 1024)
        self.SSP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = conv2d(512, 256, 1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = conv2d(256, 128, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)

        self.yolo_head3 = yolo_head([256, len(anchor_mask[0])*(5+num_classes)], 128)

        self.down_sample1 = conv2d(128, 256, 3, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)

        self.yolo_head2 = yolo_head([512, len(anchor_mask[1])*(5+num_classes)], 256)

        self.down_sample2 = conv2d(256, 512, 3, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)

        self.yolo_head1 = yolo_head([1024, len(anchor_mask[2])*(5+num_classes)], 512)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)

        P5 = self.conv1(x0)
        P5 = self.SSP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4, P5_upsample], axis=1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3, P4_upsample], axis=1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], axis=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], axis=1)
        P5 = self.make_five_conv4(P5)

        #
        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)

        return out0, out1, out2








