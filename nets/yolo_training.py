#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/25 11:18
# @Author  : shiman
# @File    : yolo_training.py
# @describe:

import torch
import torch.nn as nn
from functools import partial
import math
import numpy as np

def weight_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method {init_type} is not implemented')
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print(f'initialize network with {init_type} type')

    net.apply(init_func)
    return


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    """获得学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask,
                 label_smoothing=0, focal_loss=False, alpha=0.25, gamma=2):
        super(YOLOLoss, self).__init__()

        #   13x13的特征层对应的anchor是[142, 110],[192, 243],[459, 401]
        #   26x26的特征层对应的anchor是[36, 75],[76, 55],[72, 146]
        #   52x52的特征层对应的anchor是[12, 16],[19, 36],[40, 28]

        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = num_classes + 5
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0]*input_shape[1]) / (416**2)
        self.cls_ratio = 1 * (num_classes / 80)

        self.focal_loss = focal_loss
        self.focal_loss_ratio = 10
        self.alpha = alpha
        self.gamma = gamma

        self.ignore_threshold = 0.5
        self.cuda = cuda

    def calcute_iou(self, _box_a, _box_b):
        # box (c_x, c_y, w, h)

        # 计算真实框的(x1,y1,x2,y2)
        b1_x1, b1_x2 = _box_a[:,0] - _box_a[:,2]/2,  _box_a[:,0] + _box_a[:,2]/2
        b1_y1, b1_y2 = _box_a[:,1] - _box_a[:,3]/2,  _box_a[:,1] + _box_a[:,3]/2

        # 计算预测框的(x1,y1,x2,y2)
        b2_x1, b2_x2 = _box_b[:,0] - _box_b[:,2]/2,  _box_b[:,0] + _box_b[:,2]/2
        b2_y1, b2_y2 = _box_b[:,1] - _box_b[:,3]/2,  _box_b[:,1] + _box_b[:,3]/2

        # 将真实框和预测框都转换称左上角右下角的形式
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:,0], box_a[:,1], box_a[:,2], box_a[:,3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:,0], box_b[:,1], box_b[:,2], box_b[:,3] = b2_x1, b2_y1, b2_x2, b2_y2

        # 真实框数量，预测框数量
        A, B = box_a.size(0), box_b.size(0)

        # 计算相交面积
        max_xy = torch.min(box_a[:,2:].unsqueeze(1).expand(A,B,2), box_b[:,2:].unsqueeze(0).expand(A,B,2))
        min_xy = torch.max(box_a[:,:2].unsqueeze(1).expand(A,B,2), box_b[:,:2].unsqueeze(0).expand(A,B,2))
        inter = torch.clamp((max_xy-min_xy), min=0)
        inter = inter[:,:,0] * inter[:,:,1]
        # 计算各自框的面积  # [A,B]
        area_a = (_box_a[:,2]*_box_a[:,3]).unsqueeze(1).expand_as(inter)
        area_b = (_box_b[:,2]*_box_b[:,3]).unsqueeze(0).expand_as(inter)
        #
        union = area_a+area_b - inter
        return inter/union  #[A,B]

    def get_target(self, l, targets, anchors, in_h, in_w):
        """获取每幅影像上每个真实框最对应的一个先验框的信息 (bs, k, i, j, 5+num_classes)"""

        bs = len(targets)
        # 选取那些先验框不包含物体
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        # 网络更加关注小目标
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        #
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)
        #
        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            # 真实框映射到特征层上位置(targets[b] shape: (num_box, box_attrs)  (c_x, c_y, w, h)
            batch_target = torch.zeros_like(targets[b])
            batch_target[:,[0,2]] = targets[b][:,[0,2]] * in_w
            batch_target[:,[1,3]] = targets[b][:,[1,3]] * in_h
            batch_target[:,4] = targets[b][:,4]
            batch_target = batch_target.cpu()

            # 真实框转换形式 (num_true_box, 4)  (0,0,tar_w,tar_h)
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0),2)),batch_target[:,[2,3]]), dim=1))
            # 先验框 (9, 4) (0,0,w,h)
            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), dim=1))
            # 计算每个真实框与九个先验框的重合情况  [num_true_box, 9]
            iou_info = self.calcute_iou(gt_box, anchor_shapes)
            # 获取每个真实框重合度最大的先验框序号
            best_ns = torch.argmax(iou_info, dim=1)

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    continue
                # 判断先验框是当前特征点的哪个先验框
                k = self.anchors_mask[l].index(best_n)
                # 获取真实框属于哪个网格
                i = torch.floor(batch_target[t,0]).long()
                j = torch.floor(batch_target[t,1]).long()
                # 真实框的种类
                c = batch_target[t,4].long()
                #  代表无目标的特征点 （0代表该特征点位于特定anchor下有目标）
                noobj_mask[b,k,j,i] = 0
                #
                y_true[b,k,j,i,0] = batch_target[t,0]
                y_true[b,k,j,i,1] = batch_target[t,1]
                y_true[b,k,j,i,2] = batch_target[t,2]
                y_true[b,k,j,i,3] = batch_target[t,3]
                y_true[b,k,j,i,4] = 1
                y_true[b,k,j,i,5+c] = 1
                # 获取xywh的比例， 大目标loss权重小，小目标loss权重大
                box_loss_scale[b,k,j,i] = batch_target[t,2]*batch_target[t,3] /in_w / in_h

        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, w, h, targets, scaled_anchors, in_h, in_w, noobj_mask):
        """ x, y, h, w 是先验框的中心，宽高的调整参数值 shape (bs, 3, in_h, in_w)
            :return: 更新负样本noobj_mask， 先验框与预测框中重合度大的不适合作为负样本
                     预测框preds_boxes， 先验框整合先验框的调整参数

        """
        bs = len(targets)

        # 生成网格，先验框中心，网格左上角(bs, 3, in_h, in_w)
        grid_x = torch.linspace(0,in_w-1,in_w).repeat(in_h,1).repeat(int(bs*len(self.anchors_mask[l])),1,1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0,in_h-1,in_h).repeat(in_w,1).t().repeat(int(bs*len(self.anchors_mask[l])),1,1).view(y.shape).type_as(x)

        #  生成先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)
        #
        anchor_w = anchor_w.repeat(bs,1).repeat(1,1,in_h,in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs,1).repeat(1,1,in_h,in_w).view(h.shape)
        # 计算调整后的先验框的中心和宽高
        pred_boxes_x = torch.unsqueeze(x+grid_x,-1)
        pred_boxes_y = torch.unsqueeze(y+grid_y,-1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w)*anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h)*anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)

        for b in range(bs):
            # 将预测结果转换一个形式 num_anchors, 4
            pred_boxes_for_ignore = pred_boxes[b].reshape(-1,4)
            # 计算真实框， 并把真实框转换成相对于特征层的大小
            # gt_num    num_true_box, 4
            if len(targets[b]) >0:
                batch_target = torch.zeros_like(targets[b])
                # 计算出正样本在特征层上的中心点
                batch_target[:,[0,2]] = targets[b][:,[0,2]] * in_w
                batch_target[:,[1,3]] = targets[b][:,[1,3]] * in_h
                batch_target = batch_target[:,:4].type_as(x)
                # 计算交并比
                anch_ious = self.calcute_iou(batch_target, pred_boxes_for_ignore)
                #  每个先验框对应真实框的最大重合度
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])  # (3, in_h ,in_w)
                #
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes

    def box_ciou(self, b1, b2):
        """
        :param b1: tensor, shape(bs,3,h,w,4), xywh
        :param b2:
        :return:
        """
        # 预测框x1,y1,x2,y2
        b1_xy = b1[...,[0,1]]
        b1_wh = b1[...,[2,3]]
        b1_wh_half= b1_wh/2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        # 真实框x1,y1,x2,y2
        b2_xy = b2[...,[0,1]]
        b2_wh = b2[...,[2,3]]
        b2_wh_half = b2_wh/2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # 计算iou
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[...,0]*intersect_wh[...,1]
        b1_area = b1_wh[...,0]*b1_wh[...,1]
        b2_area = b2_wh[...,0]*b2_wh[...,1]
        union_area = b1_area + b2_area + intersect_area
        iou = intersect_area / union_area

        # 计算中心的差距
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy),2), dim=-1)

        # 计算包裹两个框最小框的左上角和右下角
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes-enclose_mins, torch.zeros_like(enclose_mins))

        # 计算对角线距离
        enclose_diagonal = torch.sum(torch.pow(enclose_wh,2), dim=-1)
        ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

        v = (4 / math.pi**2) * torch.pow((torch.atan(b1_wh[...,0]/torch.clamp(b1_wh[...,1],min=1e-6)) - torch.atan(b2_wh[...,0]/torch.clamp(b2_wh[...,1],min=1e-6))),2)
        alpha = v / torch.clamp((1.0-iou+v), min=1e-6)
        ciou = ciou - alpha*v
        return ciou

    def forward(self, l, input, targets=None):
        """
        :param l: 代表第几个有效特征层
        :param input:  shape: bs, 3(5+num_classes), 13, 13 (26, 26 | 52, 52)
        :param targets: 真实框标签情况 [bs, num_gt, 5]
        :return:
        """

        #  图片数量和特征层高、宽
        bs, _, in_h, in_w = input.shape
        #  计算步长  一个特征点对应的原图片上的多少像素点(32、16、8)
        stride_h, stride_w = self.input_shape[0]/in_h, self.input_shape[1]/in_w
        #  获取特征层上的anchor大小（原始anchor针对的是input_shape图像大小）
        scaled_anchors = [(a_w/stride_w, a_h/stride_h) for a_w, a_h in self.anchors]
        #  (bs, 3, num_classes+5, in_h, in_w) -> (bs, 3, in_h, in_w, num_classes+5)
        prediction = input.reshape(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0,1,3,4,2).contiguous()
        #  先验框中心调整参数
        x = torch.sigmoid(prediction[...,0])
        y = torch.sigmoid(prediction[...,1])
        #  先验框宽、高调整参数
        w = prediction[...,2]
        h = prediction[...,3]
        conf = torch.sigmoid(prediction[...,4])
        pred_cls = torch.sigmoid(prediction[...,5:])
        #  获取网络应该有的预测结果（真实值）
        # y_true shape (bs,3,in_h,in_w,5+num_classes)
        y_true, noobj_mask, box_loss_scale = self.get_target(l,targets, self.anchors, in_h, in_w)
        # 将预测结果进行解码，判断预测结果和真实值的重合程度, 并获得调整后的预测框
        # pred_boxes shape(bs,3,in_h,in_w,4) xywh
        noobj_mask, pred_boxes = self.get_ignore(l, x,y,w,h,targets, scaled_anchors, in_h, in_w, noobj_mask)
        if self.cuda:
            y_true = y_true.type_as(x)
            noobj_mask = noobj_mask.type_as(x)
            box_loss_scale = box_loss_scale.type_as(x)
        # 2-宽高的乘积代表真实框越大，比重越小，小框的比重更大，使用iou损失时，大中小目标的回归损失不存在比例失衡问题，故弃用
        box_loss_scale = 2 - box_loss_scale
        #
        loss = 0
        obj_mask = y_true[...,4] == 1
        n = torch.sum(obj_mask)
        if n !=0:
            # 计算预测结果和真实结果的差距  (bs, 3, in_h, in_w)
            ciou = self.box_ciou(pred_boxes, y_true[...,:4]).type_as(x)
            loss_loc = torch.mean((1-ciou)[obj_mask])

            loss_cls = torch.mean(self.BCELoss(pred_cls[obj_mask], y_true[...,5:][obj_mask]))
            loss += loss_loc*self.box_ratio+ loss_cls*self.cls_ratio

        #   计算是否包含物体的置信度损失
        if self.focal_loss:
            pos_neg_ratio   = torch.where(obj_mask, torch.ones_like(conf) * self.alpha, torch.ones_like(conf) * (1 - self.alpha))
            hard_easy_ratio = torch.where(obj_mask, torch.ones_like(conf) - conf, conf) ** self.gamma
            loss_conf   = torch.mean((self.BCELoss(conf, obj_mask.type_as(conf)) * pos_neg_ratio * hard_easy_ratio)[noobj_mask.bool() | obj_mask]) * self.focal_loss_ratio
        else:
            loss_conf   = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])

        loss += loss_conf*self.balance[l]*self.obj_ratio

        return loss

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = torch.clamp(t, min=t_min, max=t_max)
        return result

    def BCELoss(self, pred, target):
        """
        有物体的预测框和验证狂
        """
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0-epsilon)
        output = - target * torch.log(pred) - (1.0-target)*torch.log(1.0-pred)
        return output










































