#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/25 17:07
# @Author  : shiman
# @File    : utils_fit.py
# @describe:

import os
import torch
from tqdm import tqdm
from ..nets.yolo_training import get_lr

def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                  epoch_step, epoch_step_val, gen ,gen_val, Epoch, cuda, save_period, save_dir):
    loss = 0
    val_loss = 0

    print('Start Train')
    model_train.train()
    with tqdm(total=epoch_step, desc=f'Epoch {epoch+1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()
                    targets = [ann.cuda() for ann in targets]

            #
            optimizer.zero_grad()
            outputs = model_train(images)
            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()

            pbar.set_postfix(**{'loss': loss / (iteration+1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    print('Start Val')
    model_train.eval()
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch+1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration > epoch_step_val:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()
                    targets = [ann.cuda() for ann in targets]
                optimizer.zero_grad()
                outputs = model_train(images)
                loss_value_all = 0
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all += loss_item
                loss_value = loss_value_all

            val_loss += loss_value.item()

            pbar.set_postfix(**{'val_loss':val_loss / (iteration+1),})
            pbar.update(1)

    loss_history.append_loss(epoch+1, loss/epoch_step, val_loss/epoch_step)
    print(f'Epoch:{epoch+1}/{Epoch}')
    print(f'Total Loss:{loss/epoch_step:.3f} || Val Loss:{val_loss/epoch_step_val:.3f}')
    if (epoch + 1) // save_period == 0 or epoch+1 == Epoch:
        torch.save(model.state_dict(), f'{save_dir}/epoch{epoch+1}-loss{loss/epoch_step:.3f}-val_loss{val_loss/epoch_step_val:.3f}')






