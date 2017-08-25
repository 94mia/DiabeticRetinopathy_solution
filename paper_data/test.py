# import torch.nn as nn
# import torchvision
#
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import json
# import pandas as pd
# import numpy as np
#
# import os
#
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
#
# class BinClsDataSet(torch.utils.data.Dataset):
#     def __init__(self, root, config, crop_size, scale_size, baseline=False):
#         super(BinClsDataSet, self).__init__()
#         self.root = root
#         self.config = config
#         self.crop_size = crop_size
#         self.scale_size = scale_size
#         df = pd.DataFrame.from_csv(config)
#         self.images_list = []
#         for index, row in df.iterrows():
#             self.images_list.append(row)
#         with open('info.json', 'r') as fp:
#             info = json.load(fp)
#         mean_values = torch.from_numpy(np.array(info['mean'], dtype=np.float32) / 255)
#         std_values = torch.from_numpy(np.array(info['std'], dtype=np.float32) / 255)
#         eigen_values = torch.from_numpy(np.array(info['eigval'], dtype=np.float32))
#         eigen_vectors = torch.from_numpy(np.array(info['eigvec'], dtype=np.float32))
#         if baseline:
#             self.transform = transforms.Compose([
#                 transforms.RandomCrop(crop_size),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=mean_values, std=std_values),
#             ])
#
#     def __getitem__(self, item):
#         return self.transform(Image.open(os.path.join(self.root, self.images_list[item][0]+'_'+str(self.scale_size)+'.png'))), self.images_list[item][2]
#
#     def __len__(self):
#         return len(self.images_list)
#
#
# root = '/home/weidong/code/dr/DiabeticRetinopathy_solution/data/tmp/512'
# config = '/home/weidong/code/dr/DiabeticRetinopathy_solution/data/tmp/flags/512.csv'
# crop_size = 224
# scale_size = 512
#
# dataset_train = DataLoader(BinClsDataSet(root, config, crop_size, scale_size, True), batch_size=10, num_workers=1,
#                              shuffle=True, pin_memory=True)
#
# model = torchvision.models.densenet121(True)
#
# model = torch.nn.DataParallel(model).cuda()
#
# for index, (image,label) in enumerate(dataset_train):
#     output = model(Variable(image.cuda()))
#     print(output)
#
# print(model)






import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F

import torch.nn as nn
import torchvision

import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import pandas as pd
import numpy as np

import os

from torch.utils.data import DataLoader
from torch.autograd import Variable

class multi_task_model(nn.Module):
    def __init__(self, name, inmap, classes_multi, weights=None, scratch=False):
        super(multi_task_model, self).__init__()
        self.base = None
        self.cls = None
        self.featmap = 7
        self.cls = [None]*len(classes_multi)
        if name == 'rsn101':
            base_model = torchvision.models.resnet101()
            self.featmap = inmap // 32
            self.base = nn.Sequential(*list(base_model.children())[:-2])
            self.cls0 = nn.Linear(2048, classes_multi[0])
            self.cls1 = nn.Linear(2048, classes_multi[1])
            if len(classes_multi) == 3:
                self.cls2 = nn.Linear(2048, classes_multi[1])
            # self.cls = nn.Linear(2048, classes_multi)
        elif name == 'dsn121':
            base_model = torchvision.models.densenet121()
            self.featmap = inmap // 32
            self.base = list(base_model.children())[0]
            self.cls = [nn.Linear(base_model.classifier.in_features, classes) for classes in classes_multi]
        if not scratch:
            base_model.load_state_dict(torch.load('../pretrained/'+name+'.pth'))
        if weights:
            self.load_state_dict(torch.load(weights))

    def forward(self, x):
        feature = self.base(x)
        out = F.relu(feature, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.featmap).view(feature.size(0),-1)
        out1 = self.cls0(out)
        out2 = self.cls1(out)
        return out1, out2

class BinClsDataSet(torch.utils.data.Dataset):
    def __init__(self, root, config, crop_size, scale_size, baseline=False):
        super(BinClsDataSet, self).__init__()
        self.root = root
        self.config = config
        self.crop_size = crop_size
        self.scale_size = scale_size
        df = pd.DataFrame.from_csv(config)
        self.images_list = []
        for index, row in df.iterrows():
            self.images_list.append(row)
        with open('info.json', 'r') as fp:
            info = json.load(fp)
        mean_values = torch.from_numpy(np.array(info['mean'], dtype=np.float32) / 255)
        std_values = torch.from_numpy(np.array(info['std'], dtype=np.float32) / 255)
        eigen_values = torch.from_numpy(np.array(info['eigval'], dtype=np.float32))
        eigen_vectors = torch.from_numpy(np.array(info['eigvec'], dtype=np.float32))
        if baseline:
            self.transform = transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_values, std=std_values),
            ])

    def __getitem__(self, item):
        return self.transform(Image.open(os.path.join(self.root, self.images_list[item][0]+'_'+str(self.scale_size)+'.png'))), self.images_list[item][2]

    def __len__(self):
        return len(self.images_list)


root = '/home/weidong/code/dr/DiabeticRetinopathy_solution/data/tmp/512'
config = '/home/weidong/code/dr/DiabeticRetinopathy_solution/data/tmp/flags/512.csv'
crop_size = 224
scale_size = 512

dataset_train = DataLoader(BinClsDataSet(root, config, crop_size, scale_size, True), batch_size=1, num_workers=1,
                             shuffle=True, pin_memory=True)

model = multi_task_model('rsn101', 224, [5,4])
# model = multi_task_model('rsn101', 224, 5)


model = torch.nn.DataParallel(model).cuda()

for index, (image,label) in enumerate(dataset_train):
    # model = torch.nn.DataParallel(model).cuda()
    output1, output2 = model(Variable(image.cuda()))
    print(output1)
    print(output2)