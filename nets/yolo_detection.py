#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 18:20
# @Author  : shiman
# @File    : yolo_detection.py
# @describe:


import numpy as np
import torch
import torch.nn as nn

from ..utils.utils import get_classes, get_anchors, cvtColor, resize_image, \
                            preprocess_input, draw_detection_image
from ..utils.utils_bbox import DecodeBox
from .yolo4 import YoloBody

class YOLO(object):
    _defaults = {
        'model_path':'data/yolo4_weights.pth',
        'classes_path':'data/coco_classes.txt',
        'anchors_path':'data/yolo_anchors.txt',
        'anchors_mask': [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        'input_shape': [416,416],
        'confidence':0.5,
        'nms_iou':0.3,
        'letterbox_image':False,
    }

    @classmethod
    def get_defaults(cls, k):
        if k in cls._defaults:
            return cls._defaults[k]
        else:
            return f'Unrecognized attribute name {k}'

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for k, v in kwargs.items():
            setattr(self, k, v)

        # 获得种类和先验框数量
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, self.input_shape)

        self.generate()

    def generate(self, onnx=False):
        self.net = YoloBody(self.anchors_mask, self.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print(f'{self.model_path} model, anchors and classes loaded')

        if not onnx:
            if torch.cuda.is_available():
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, crop=False, count=False):
        # 获取影像宽、高
        image_shape = np.array(np.shape(image)[0:2])
        # 转成RGB
        image = cvtColor(image)
        # 图像重采样resize
        image_data = resize_image(image, self.input_shape, letterbox_image=self.letterbox_image)
        # 添加bs维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2,0,1)), axis=0)
        #
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if torch.cuda.is_available():
                images = images.cuda()
            #
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
            if results[0] is None:
                return image

            #
            draw_detection_image(image, results[0], self.classes_path, self.input_shape)
            #
            if count:
                top_label = np.array(results[0][:, 6], dtype='int32')
                top_conf = results[0][:, 4] * results[0][:, 5]
                top_boxes = results[0][:, :4]

                print('top_lable:',top_label)
                classes_nums = np.zeros([self.num_classes])
                for i in range(self.num_classes):
                    num = np.sum(top_label == i)
                    if num > 0:
                        print(f'{self.class_names[i]}: {num}')
                    classes_nums[i] = num
                print(f'classes_nums: {classes_nums}')

            if crop:
                pass
            return image
