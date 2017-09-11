
import torch
import json
import numpy as np
import torchvision.transforms as transforms

from utils import PILColorJitter, Lighting

from PIL import Image

import pandas as pd
import os

from torch.utils.data import DataLoader

import torchvision

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

from utils import quadratic_weighted_kappa, kappa_confusion_matrix, AverageMeter

import time
import math

import torch.backends.cudnn as cudnn

from sklearn.metrics import confusion_matrix


class SingleChannelClsDataSet(torch.utils.data.Dataset):
    def __init__(self, root, root_ahe, config, crop_size, scale_size, baseline=False):
        super(SingleChannelClsDataSet, self).__init__()
        self.root = root
        self.root_ahe = root_ahe
        self.config = config
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.baseline = baseline
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
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                PILColorJitter(),
                transforms.ToTensor(),
                Lighting(alphastd=0.01, eigval=eigen_values, eigvec=eigen_values),
                transforms.Normalize(mean=mean_values, std=std_values),
            ])

        self.transform_ahe = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values, std=std_values),
        ])

    def __getitem__(self, item):
        return self.transform(
            Image.open(os.path.join(self.root, self.images_list[item][0] + '_' + str(self.scale_size) + '.png'))), \
               self.images_list[item][1], self.images_list[item][2], self.images_list[item][3]

    def __len__(self):
        return len(self.images_list)

class SingleChannelClsValDataSet(torch.utils.data.Dataset):
    def __init__(self, root, root_ahe, config, crop_size, scale_size, baseline=False):
        super(SingleChannelClsValDataSet, self).__init__()
        self.root = root
        self.root_ahe = root_ahe
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
        self.transform = transforms.Compose([
            transforms.Scale(self.scale_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values, std=std_values),
        ])
        self.transform_ahe = self.transform

    def __getitem__(self, item):
        return self.transform(Image.open(os.path.join(self.root, self.images_list[item][0]+'_'+str(self.scale_size)+'.png'))), \
               self.images_list[item][1], self.images_list[item][2], self.images_list[item][3]

    def __len__(self):
        return len(self.images_list)

def initialize_cls_weights(cls):
	for m in cls.modules():
		if isinstance(m, nn.Conv2d):
			n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			m.weight.data.normal_(0, math.sqrt(2. / n))
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1)
			m.bias.data.zero_()
		elif isinstance(m, nn.Linear):
			m.weight.data.normal_(0, 0.01)
			m.bias.data.zero_()

class single_channel_model(nn.Module):
    def __init__(self, name, inmap, multi_classes, weights=None, scratch=False, outbincls=True):
        super(single_channel_model, self).__init__()
        self.name = name
        self.weights = weights
        self.inmap = inmap
        self.multi_classes = multi_classes
        self.cls0 = None
        self.cls1 = None
        self.cls2 = None
        self.featmap = inmap // 32
        self.planes = 2048
        self.outbincls = outbincls
        base_model = None
        if name == 'rsn18':
            base_model = torchvision.models.resnet18()
            self.planes = 512
        elif name == 'rsn34':
            base_model = torchvision.models.resnet34()
            self.planes = 512
        elif name == 'rsn50':
            base_model = torchvision.models.resnet50()
            self.planes = 2048
        elif name == 'rsn101':
            base_model = torchvision.models.resnet101()
            self.planes = 2048
        elif name == 'rsn152':
            base_model = torchvision.models.resnet152()
            self.planes = 2048
        elif name == 'dsn121':
            base_model = torchvision.models.densenet121()
            self.planes = base_model.classifier.in_features
        elif name == 'dsn161':
            base_model = torchvision.models.densenet161()
            self.planes = base_model.classifier.in_features
        elif name == 'dsn169':
            base_model = torchvision.models.densenet169()
            self.planes = base_model.classifier.in_features
        elif name == 'dsn201':
            base_model = torchvision.models.densenet201()
            self.planes = base_model.classifier.in_features

        if not scratch:
            base_model.load_state_dict(torch.load('../pretrained/'+name+'.pth'))

        self.base = nn.Sequential(*list(base_model.children())[:-2])
        if name == 'rsn18' or name == 'rsn34' or name == 'rsn50' or name == 'rsn101' or name == 'rsn152':
            self.base = nn.Sequential(*list(base_model.children())[:-2])
        elif name == 'dsn121' or name == 'dsn161' or name == 'dsn169' or name == 'dsn201':
            self.base = list(base_model.children())[0]

        self.cls0 = nn.Linear(self.planes, multi_classes[0])
        self.cls1 = nn.Linear(self.planes, multi_classes[1])
        self.cls2 = nn.Linear(self.planes, 2)

        initialize_cls_weights(self.cls0)
        initialize_cls_weights(self.cls1)
        initialize_cls_weights(self.cls2)
        if weights:
            self.load_state_dict(torch.load(weights))

    def forward(self, x):
        feature = self.base(x)
        # when 'inplace=True', some errors occur!!!!!!!!!!!!!!!!!!!!!!
        out = F.relu(feature, inplace=False)
        out = F.avg_pool2d(out, kernel_size=self.featmap).view(feature.size(0), -1)
        out1 = self.cls0(out)
        out2 = self.cls1(out)
        out3 = self.cls2(out)
        if self.outbincls:
            return out1, out2, out3
        else:
            return out1, out2