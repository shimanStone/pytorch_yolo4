#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 16:13
# @Author  : shiman
# @File    : utils_bbox.py
# @describe:

import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np

class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape,
                 anchor_mask = [[6,7,8],[3,4,5],[0,1,2]]):
        super(DecodeBox, self).__init__()

        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5+num_classes
        self.input_shape = input_shape
        # 13x13的特征层对应的anchor是[142, 110],[192, 243],[459, 401]
        # 26x26的特征层对应的anchor是[36, 75],[76, 55],[72, 146]
        # 52x52的特征层对应的anchor是[12, 16],[19, 36],[40, 28]
        self.anchor_mask = anchor_mask

    def decode_box(self, inputs):
        outputs = []
        # inputs = [bs,3*85,13,13], [bs,3*85,26,26], [bs,3*85,52,52]
        for i , input in enumerate(inputs):
            batch_size, _, input_height, input_width = input.size()
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            # 获得相对于特征层的scaled_anchor
            scaled_anchors = [(anchor_width/stride_w, anchor_height/stride_h) for anchor_width, anchor_height in self.anchors[self.anchor_mask[i]]]

            prediction = input.view(batch_size,len(self.anchor_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0,1,3,4,2).contiguous()
            # 先验框中心位置调整
            x = torch.sigmoid(prediction[...,0])
            y = torch.sigmoid(prediction[...,1])
            w, h = prediction[...,2], prediction[...,3]
            # 是否有物体置信度
            conf = torch.sigmoid(prediction[...,4])
            # 种类置信度
            pred_cls = torch.sigmoid(prediction[...,5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # 生成先验框网格中心(bs,3,13,13)
            grid_x = torch.linspace(0, input_width-1, input_width).repeat(input_height,1).repeat(
                batch_size*len(self.anchor_mask[i]),1,1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height-1, input_height).repeat(input_width,1).t().repeat(
                batch_size*len(self.anchor_mask[i]),1,1).view(y.shape).type(FloatTensor)

            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size,1).repeat(1,1,input_height*input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size,1).repeat(1,1,input_height*input_width).view(h.shape)

            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

            #   将输出结果归一化成小数的形式
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)

        return  outputs

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        box_yx = box_xy[...,::-1]
        box_hw = box_wh[...,::-1]
        input_shape, image_shape = np.array(input_shape), np.array(image_shape)
        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset = (input_shape - new_shape) / 2 / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) *scale
            box_hw *= scale

        box_mins = box_yx - (box_hw/2.)
        box_maxes = box_yx + (box_hw/2.)

        boxes = np.concatenate([box_mins[...,0:1],box_mins[...,1:2],box_maxes[...,0:1],box_maxes[...,1:2]], axis=-1)
        boxes *= np.concatenate([image_shape,image_shape], axis=-1)

        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image,
                            conf_thres=0.5, nms_thres=0.4):
        # 将预测结果转换称左上右下角形式
        box_corner = prediction.new(prediction.shape)
        box_corner[:,:,0] = prediction[:,:,0] - prediction[:,:,2]/2
        box_corner[:,:,1] = prediction[:,:,1] - prediction[:,:,3]/2
        box_corner[:,:,2] = prediction[:,:,0] + prediction[:,:,2]/2
        box_corner[:,:,3] = prediction[:,:,1] + prediction[:,:,3]/2
        prediction[:,:,:4] = box_corner[:,:,:4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # 最大分类置信度及索引
            class_conf, class_pred = torch.max(image_pred[:,5:5+num_classes], 1, keepdim=True)
            # 利用置信度进行第一轮筛选(是否包含物体image_conf * 概率最高的物体的概率class_conf)
            conf_mask = (image_pred[:,4]*class_conf[:,0] >= conf_thres).squeeze()
            # 利用conf进行预测结果筛选
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            #
            if not image_pred.size(0):
                continue
            # x1,y1,x2,y2,obj_conf,class_conf,class_pred (shape:num_anchors,7)
            detections = torch.cat((image_pred[:,:5], class_conf.float(), class_pred.float()), 1)
            #
            unique_labels = detections[:,-1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                # 获取某一类得分筛选后全部预测结果
                detections_class = detections[detections[:,-1] == c]
                #
                keep = nms(detections_class[:,:4],detections_class[:,4]*detections_class[:,5],nms_thres)
                #
                max_detections = detections_class[keep]

                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections), 0)

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:,0:2] + output[i][:,2:4])/2,  output[i][:,2:4] - output[i][:,0:2]
                output[i][:,0:4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return output








