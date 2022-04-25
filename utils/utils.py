#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 18:06
# @Author  : shiman
# @File    : utils.py
# @describe:

import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_anchors(anchors_path):
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()

    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1,2)
    return anchors, len(anchors)

def cvtColor(image):
    if len(np.shape(image)) == 3 or np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def resize_image(image, out_size, letterbox_image):
    iw, ih = image.size
    w, h = out_size
    if letterbox_image:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', out_size, (128,128,128))
        new_image.paste(image, ((w-nw)//2,(h-nh)//2))
    else:
        new_image = image.resize((w,h), Image.BICUBIC)

    return new_image

def preprocess_input(image):
    image /= 255.0
    return image

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return


def draw_detection_image(img_path, bbox_arr, classes_path, input_shape, font=r'E:\ml_code\data\frcnn\simhei.ttf'):
    if isinstance(img_path, str):
        image = Image.open(img_path)
    else:
        image = img_path

    # 获得分类信息
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    # 分配颜色
    hsv_tuple = [(x / len(class_names), 1, 1) for x in range(len(class_names))]
    rgb_color = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuple))
    rgb_color = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rgb_color))

    # 设置字体样式及边框厚度
    font = ImageFont.truetype(font=font, size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))

    # 绘制每个预测框
    for i, info in enumerate(bbox_arr):
        # 获取标签名，四角范围，conf
        label_index = int(info[6])
        label_name = class_names[label_index]
        box = info[0:4]
        score = info[4] * info[5]
        # 规范四角范围
        top, left, bottom, right = box
        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom).astype('int32'))
        right = min(image.size[0], np.floor(right).astype('int32'))
        # 标签文本
        label = f'{label_name}{score:.2f}'
        # 开始画图
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)  # (label_w_size, label_h_size)
        # 设置 标签文字起始位置坐标
        if top > label_size[1]:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        # 绘框
        draw.rectangle([left, top, right, bottom], outline=rgb_color[label_index], width=thickness)
        # 绘制文本框
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=rgb_color[label_index])
        # 绘制文本
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return image

