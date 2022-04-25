#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 17:17
# @Author  : shiman
# @File    : predict.py
# @describe:


from PIL import Image

from pytorch_yolo4.nets.yolo_detection import YOLO


if __name__ == '__main__':

    yolo = YOLO()

    mode = 'predict'
    crop=False
    count = False

    img = r'E:\ml_code\pytorch_yolo4\data\street.jpg'
    image = Image.open(img)
    r_image = yolo.detect_image(image, crop=crop, count=count)
    r_image.show()
