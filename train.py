#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/25 10:43
# @Author  : shiman
# @File    : train.py
# @describe:

import os
root_dir = os.path.dirname(os.path.abspath(__file__))

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from pytorch_yolo4.utils.utils import get_classes, get_anchors
from pytorch_yolo4.utils.callbacks import LossHistory
from pytorch_yolo4.nets.yolo4 import YoloBody
from pytorch_yolo4.nets.yolo_training import weight_init, YOLOLoss, get_lr_scheduler, set_optimizer_lr
from pytorch_yolo4.utils.dataloader import YoloDataset, yolo_dataset_collate
from pytorch_yolo4.utils.utils_fit import fit_one_epoch

if __name__ == '__main__':

    Cuda = torch.cuda.is_available()
    # 分类文件
    classes_path = f'{root_dir}/data/voc_classes.txt'
    # anchors配置
    anchors_path = f'{root_dir}/data/yolo_anchors.txt'
    anchors_mask = [[6,7,8],[3,4,5],[0,1,2]]
    # 模型信息
    model_path = f'{root_dir}/data/yolo4_voc_weights.pth'
    input_shape = [416, 416]
    #
    pretrained = False
    #
    mosaic = True
    label_smoothing = 0

    # 冻结阶段参数
    init_epoch = 0
    freeze_epoch = 50
    freeze_batch_size = 8

    # 解冻阶段参数
    unfreeze_epoch = 100
    unfreeze_batch_size = 4

    # 是否进行冻结训练
    freeze_train = True

    # 模型的最大和最小学习率
    init_lr = 1e-2
    min_lr = init_lr * 0.01

    # 优化器配置
    optimizer_type = 'sgd'
    momentum = 0.937
    weight_decay = 5e-4

    #
    lr_decay_type = 'cos'

    #
    focal_loss = False  # 是否使用Focal loss平衡正负样本
    focal_alpha = 0.25  # Focal loss的正负样本平衡参数
    focal_gamma = 2     # Focal loss的难易分类样本平衡参数
    #
    save_period = 1     # 多少epoch保存一次权重
    #
    save_dir = 'logs'
    #
    num_workers = 4   # 设置是否多线程读取数据
    #
    train_annotation_path, val_annotation_path = \
        f'{root_dir}/data/2007_train.txt', f'{root_dir}/data/2007_val.txt'
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0

    # 获取分类及先验框
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    # 创建yolo模型
    model = YoloBody(anchors_mask, num_classes, pretrained=pretrained)
    if not pretrained:
        weight_init(model)

    if model_path != '':
        print(f'load weights {model_path}')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # 损失函数
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask,
                         label_smoothing, focal_loss, focal_alpha, focal_gamma)
    loss_history = LossHistory(save_dir, model, input_shape)

    #
    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # train val 数据读取
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train, num_val = len(train_lines), len(val_lines)

    #
    if True:
        unfreeze_flag = False

        if freeze_train:
            for param in model.backbone.parameters():
                param.requires_grad = False
        batch_size = freeze_batch_size if freeze_train else unfreeze_batch_size

        # 判断bs,自适应调整学习率
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type in ['adam', 'adamw'] else 5e-2
        lr_limit_min = 3e-4 if optimizer_type in ['adam', 'adamw'] else 5e-4
        init_lr_fit = min(max(batch_size/nbs*init_lr, lr_limit_min), lr_limit_max)
        min_lr_fit = min(max(batch_size/nbs*min_lr, lr_limit_min*1e-2), lr_limit_max*1e-2)

        # 根据optimizer_type 选择优化器
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
                pg0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adma': optim.Adam(pg0, init_lr_fit, betas=(momentum, 0.999)),
            'admaw': optim.AdamW(pg0, init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({'params':pg1, 'weight_decay': weight_decay})
        optimizer.add_param_group({'params':pg2})

        # 获取学习率下降公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, total_iters=unfreeze_epoch)

        # 判断每一个epoch的长度
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法继续进行训练，请扩充数据集')

        #
        train_dataset = YoloDataset(train_lines, input_shape, num_classes, unfreeze_epoch, mosaic, True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, unfreeze_epoch, False, False)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

        # 开始模型训练
        for epoch in range(init_epoch, unfreeze_epoch):

            # 模型有冻结训练部分，则解冻并设置解冻后参数(主要是由于 batch_size不同引发的参数变化)
            if epoch >= freeze_epoch and not unfreeze_flag and freeze_train:
                batch_size = unfreeze_batch_size
                # 判断bs,自适应调整学习率
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type in ['adam', 'adamw'] else 5e-2
                lr_limit_min = 3e-4 if optimizer_type in ['adam', 'adamw'] else 5e-4
                init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
                min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                # 获取学习率下降公式
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, total_iters=unfreeze_epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val ==0:
                    raise ValueError(f'数据集过小， 无法继续训练，请扩充数据集')

                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

                unfreeze_flag = True

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step,
                          epoch_step_val, gen, gen_val, unfreeze_epoch, Cuda, save_period, save_dir)
















