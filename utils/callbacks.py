#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/25 13:54
# @Author  : shiman
# @File    : callbacks.py
# @describe:


import os
import datetime
import torch
import scipy.signal
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str = datetime.datetime.strftime(datetime.datetime.now(),
                                              '%Y_%m_%d_%H_%M_%S')
        self.log_dir = f'{log_dir}/loss_{time_str}'
        self.losses = []
        self.val_losses = []

        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def add_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_losses.append(val_loss)

        with open(f'{self.log_dir}/epoch_loss.txt', 'a') as f:
            f.write(str(loss))
            f.write('\n')
        with open(f'{self.log_dir}/epoch_val_loss.txt', 'a') as f:
            f.write(str(val_loss))
            f.write('\n')

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot(epoch)

    def loss_plot(self, epoch):
        iters = range(len(self.losses))
        iters = [i+epoch for i in iters]

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_losses, 'coral', linewidth=2, label='val loss')

        try:
            num = 5 if len(self.losses) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green',
                     linestyle='--', linewidth=2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_losses, num, 3), '#8B4513',
                     linestyle='--', linewidth=2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.savefig(f'{self.log_dir}/ epoch_loss.png')

        plt.cla()
        plt.close('all')



