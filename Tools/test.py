import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import sys

sys.path.append('/home/kuangjian/workspace/CODE/Video/')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
from Data import input_data
import torchvision
from torchvision import datasets, transforms
import Data.Dataset as DD
from Models.CAT import CAT
from Models.i3d_fin import InceptionI3d
from sklearn.metrics import classification_report

def run(mode='rgb', save_model=''):
    i3d1 = InceptionI3d(400, in_channels=3)
    i3d2 = InceptionI3d(400, in_channels=3)
    cat = CAT()

    i3d1.load_state_dict(
        torch.load('path/cam1.pt')
    )
    i3d2.load_state_dict(
        torch.load('path/cam2.pt')
    )

    cat.load_state_dict(torch.load('path/checkpoint.pt'))


    i3d1.cuda()
    i3d1 = nn.DataParallel(i3d1)
    i3d2.cuda()
    i3d2 = nn.DataParallel(i3d2)
    cat.cuda()
    cat = nn.DataParallel(cat)

    i3d1.eval()
    i3d2.eval()
    cat.eval()

    st = 0
    batch_start = 0
    filename = 'data.list'

    total_t = 0.0
    correct_prediction = 0.0

    total = 0.0

    cam1_path, cam2_path, labels = input_data.get_pic(file_path=filename, clips_num=32, interval=1,
                                                      double=True, test=True, batch_index=batch_start,
                                                      st=st,cam1='cam1',cam2='cam2')

    dr = DD.DriveTestDataSet(
        cam1_path=cam1_path,
        cam2_path=cam2_path,
        labels=labels,
        double=True,
        test=True
    )

    dataloader = DD.get_test_dataloader(dr, batch_size=1)


    pre = []
    tru = []

    for data in dataloader:
        inputs1, inputs2, labels = data
        inputs1 = Variable(inputs1.cuda())
        inputs2 = Variable(inputs2.cuda())
        labels = Variable(labels.cuda())

        start = time.time()
        per_frame_logits1 = i3d1(inputs1)
        per_frame_logits2 = i3d2(inputs2)
        per_frame_logits = cat(per_frame_logits1, per_frame_logits2)
        # per_frame_logits = torch.mean(per_frame_logits, dim=2)
        end = time.time() - start
        _, per_frame_logits = torch.max(per_frame_logits.data, 1)

        print("标签:",labels)
        print("预测:",per_frame_logits)

        pre.extend(per_frame_logits.cpu())
        tru.extend(labels.cpu())

        correct_prediction += (per_frame_logits == labels).sum().item()
        total += labels.size(0)

        total_t = total_t + end
    acc = correct_prediction / total

    ave_time = total_t / total


    print(' test ACC: {:.4f}  Time: {}  AVE Time: {}   TOTAL:{}  Right:{}'.format(
        acc, total_t, ave_time,total ,correct_prediction))
    print(classification_report(tru, pre,
                                target_names=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12',
                                              'L13', 'L14', 'L15', 'L16']))

if __name__ == '__main__':
    # need to add argparse

    run(mode='rgb',save_model='path/checkpoint.pt')