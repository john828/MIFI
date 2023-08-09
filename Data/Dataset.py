import torch
import sys
sys.path.append('/home/kuangjian/workspace/CODE/Video/')
import input_data
from torch.utils import data
from torchvision import transforms
import PIL.Image as Image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import cv2 as cv
import numpy as np
class DriveDataSet(data.Dataset):
    def __init__(
        self,
        filename,
        clips_num: int=3,
        interval: int=1,
        flag = 'train',
        cam1 = "cam1",
        cam2 = "cam2"
    ):

        cam1_path,cam2_path,labels = input_data.get_pic(filename,clips_num=clips_num,interval=interval,cam1=cam1,cam2=cam2)
        self.cam1_path = cam1_path
        self.cam2_path = cam2_path
        self.labels = labels

        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[0.229, 0.224, 0.225]),
            # transforms.Normalize( mean=[0.046468433, 0.046468433, 0.046468433], std=[0.051598676, 0.051598676, 0.051598676]),
            transforms.RandomErasing()
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.046468433, 0.046468433, 0.046468433],
            #                      std=[0.051598676, 0.051598676, 0.051598676]),
        ])

        if flag == 'train':
            self.transforms = transform_train
        else:
            self.transforms = transform_test

    def __getitem__(self,index):
        cam1_paths = self.cam1_path[index]
        cam2_paths = self.cam2_path[index]

        cam1_datas = []
        cam2_datas = []
        for i in range(0, len(cam1_paths)):
            cam1_data = Image.open(cam1_paths[i])
            cam1_data = self.transforms(cam1_data)
            ##########tiff#########
            # cam1_data = cv.imread(cam1_paths[i])
            # img1_tensor = torch.from_numpy(cam1_data).float()
            # img1_tensor = img1_tensor.transpose(0, 2)
            # img1_tensor /= 255
            # img1_pil = transforms.ToPILImage()(img1_tensor).convert("RGB")
            # cam1_data = self.transforms(img1_pil)
            cam1_datas.append(cam1_data)
        cam1_datas = torch.stack(cam1_datas, dim=0).permute(1,0,2,3)

        for i in range(0, len(cam2_paths)):
            cam2_data = Image.open(cam2_paths[i])
            cam2_data = self.transforms(cam2_data)

            #####读取tiff格式#####
            # cam2_data = cv.imread(cam2_paths[i])
            # img2_tensor = torch.from_numpy(cam2_data).float()
            # img2_tensor = img2_tensor.transpose(0, 2)
            # img2_tensor /= 255
            # img2_pil = transforms.ToPILImage()(img2_tensor).convert("RGB")
            # cam2_data = self.transforms(img2_pil)

            cam2_datas.append(cam2_data)
            ##############changshi
        cam2_datas = torch.stack(cam2_datas, dim=0).permute(1,0,2,3)

        labels = self.labels[index]

        return cam1_datas,cam2_datas,labels

    def __len__(self):
        return len(self.cam1_path)


def get_training_dataloader(dataset, batch_size=4,num_workers=1, shuffle=True):
    training = dataset
    drive_training_loader = DataLoader(
        training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return drive_training_loader

def get_test_dataloader(dataset, batch_size=4,num_workers=1, shuffle=True):
    training = dataset
    drive_test_loader = DataLoader(
        training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return drive_test_loader



