import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import torchvision
from torchvision import datasets, transforms
from Data import input_data
import numpy as np
from Models.i3d_fin import InceptionI3d
from Models.CAT import CAT
from Models.loss import CASL

import math
# from new.models.Fusion.i3d_fusion import CAT
import Data.Dataset as DD


def get_train_Data():
    dr = DD.DriveDataSet(
        filename='/home/kuangjian/workspace/CODE/Video/Data/3md_train_15.list',
        interval=1,
        clips_num=32,
        flag='train'
    )
    dataloader = DD.get_training_dataloader(dr, batch_size=4)
    return dataloader


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.Conv3d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def run(init_lr=0.1, max_steps=300, save_model='', train=True):

    # setup the model

    i3d1 = InceptionI3d(400, in_channels=3)
    i3d2 = InceptionI3d(400, in_channels=3)
    cat = CAT()
    cat.apply(weight_init)
    i3d1.load_state_dict(
        torch.load('root_path/cam1.pt'), strict=False
    )
    i3d2.load_state_dict(
        torch.load('root_path/cam2.pt'), strict=False
    )

    i3d1.cuda()
    i3d1 = nn.DataParallel(i3d1)
    i3d2.cuda()
    i3d2 = nn.DataParallel(i3d2)
    cat.cuda()
    cat = nn.DataParallel(cat)

    lr = init_lr
    optimizer3 = optim.SGD(cat.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    lr3_sched = optim.lr_scheduler.MultiStepLR(optimizer3, [60, 180])

    y = 0
    l1 = 4
    l2 = 0

    print('-----最大轮次60-----')
    # (hc,neg,pos)
    loss_function = CASL(gamma_pos=l2, gamma_neg=l1, gamma_hc=y, epochs=60)
    loss_fun = nn.CrossEntropyLoss()

    # num_steps_per_update = 4  # accum gradient
    steps = 0
    best = 0.0
    n = 0
    # train it
    while steps < max_steps:  # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 30)
        start = time.time()
        # Each epoch has a training and validation phase
        if train:
            i3d1.train(True)
            i3d2.train(True)
            cat.train(True)

            dataloader = get_train_Data()

            loss_train = 0.0
            correct_prediction = 0.0
            total = 0.0
            if 60 < steps and steps <= 180:
                print('------最大轮次60--------')
                loss_function = CASL(gamma_pos=l2, gamma_neg=l1, gamma_hc=y, epochs=120)
            if steps > 180:
                print('------最大轮次100--------')
                loss_function = CASL(gamma_pos=l2, gamma_neg=l1, gamma_hc=y, epochs=120)

            for data in dataloader:
                optimizer3.zero_grad()

                inputs1, inputs2, labels = data

                inputs1 = Variable(inputs1.cuda())
                inputs2 = Variable(inputs2.cuda())
                labels = Variable(labels.cuda())

                per_frame_logits1 = i3d1(inputs1)
                per_frame_logits2 = i3d2(inputs2)

                per_frame_logits = cat(per_frame_logits1, per_frame_logits2)

                if steps <= 60:
                    loss = loss_function(per_frame_logits, labels, epoch=steps)
                    loss_train += loss.item()
                if 60 < steps and steps <= 180:
                    loss = loss_function(per_frame_logits, labels, epoch=steps - 60)
                    loss_train += loss.item()
                if steps > 180:
                    loss = loss_function(per_frame_logits, labels, epoch=steps - 180)
                    loss_train += loss.item()

                _, per_frame_logits = torch.max(per_frame_logits.data, 1)

                loss.backward()
                optimizer3.step()
                lr3_sched.step()

                correct_prediction += (per_frame_logits == labels).sum().item()
                total += labels.size(0)
            steps += 1
            n += 1

            finish = time.time()
            print('train Loss: {:.4f}  train Tot Loss: {:.4f}  train ACC: {:.4f} Time: {}'.format(loss,
                                                                                                  loss_train,
                                                                                                  correct_prediction / total,
                                                                                                  finish - start))


            if n % 5 == 0:
                train = False

        #####    val:
        if not train:
            i3d1.eval()
            i3d2.eval()
            cat.eval()
            st = 0
            batch_start = 0
            filename = 'path/3md_test.list'

            total_t = 0.0
            loss_val = 0.0
            correct_prediction = 0.0
            total = 0.0

            cam1_path, cam2_path, labels = input_data.get_pic(file_path=filename, clips_num=32, interval=1,
                                                              double=True, test=True, batch_index=batch_start,
                                                              st=st)

            dr = DD.DriveTestDataSet(
                cam1_path=cam1_path,
                cam2_path=cam2_path,
                labels=labels,
                double=True,
                test=True
            )

            dataloader = DD.get_test_dataloader(dr, batch_size=2)

            for data in dataloader:
                inputs1, inputs2, labels = data
                # print('111111111', inputs1.size())
                inputs1 = Variable(inputs1.cuda())
                inputs2 = Variable(inputs2.cuda())
                labels = Variable(labels.cuda())

                start = time.time()
                per_frame_logits1 = i3d1(inputs1)
                per_frame_logits2 = i3d2(inputs2)
                per_frame_logits = cat(per_frame_logits1, per_frame_logits2)

                # per_frame_logits = torch.mean(per_frame_logits, dim=2)
                end = time.time() - start

                loss = loss_fun(per_frame_logits, labels)
                loss_val += loss.item()

                _, per_frame_logits = torch.max(per_frame_logits.data, 1)

                correct_prediction += (per_frame_logits == labels).sum().item()
                total += labels.size(0)

                # 正确率
                # s = s + 1
                total_t = total_t + end
            acc = correct_prediction / total
            ave_time = total_t / total
            ave_loss = loss_val / total

            if acc >= best:
                torch.save(cat.module.state_dict(),
                           save_model + 'checkpoint.pt')
                best = acc

            print('val Loss: {:.4f}  val ave Loss: {:.4f}  vak ACC: {:.4f}  Time: {}  AVE Time: {} '.format(
                loss_val, ave_loss, acc, total_t, ave_time))

            train = True
            # steps += 1


if __name__ == '__main__':
    # need to add argparse

    run(mode='rgb', train=True,
        save_model='save_path/')