import sys
sys.path.append('../')
import torch
import json
import torchvision.transforms as transforms
from utils import PILColorJitter, Lighting
from PIL import Image
import os
import pandas as pd
import numpy as np

import argparse

from torch.utils.data import DataLoader

import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

import math

import torch.optim as optim

from utils import quadratic_weighted_kappa, kappa_confusion_matrix, AverageMeter
from sklearn.metrics import confusion_matrix
import time

import torch.backends.cudnn as cudnn

from loss import FocalLoss

from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description='multi-task classification options')
    parser.add_argument('--root', required=True)
    parser.add_argument('--traincsv', default=None)
    parser.add_argument('--valcsv', default=None)
    parser.add_argument('--testcsv', default=None)
    parser.add_argument('--exp', default='multi_task', help='The name of experiment')
    parser.add_argument('--batch', default=8, type=int)
    parser.add_argument('--crop', default=448, type=int)
    parser.add_argument('--size', default=512, type=int)
    parser.add_argument('--weight', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--mom', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--fix', default=100, type=int)
    parser.add_argument('--step', default=100, type=int)
    parser.add_argument('--dataset', default='kaggle', choices=['kaggle', 'zhizhen'])
    parser.add_argument('--model', default='rsn101', choices=[
        'rsn18', 'rsn34', 'rsn50', 'rsn101', 'rsn150', 'dsn121', 'dsn161', 'dsn169', 'dsn201',
    ])
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--phase', default='train', choices=['train', 'test', 'infer'])
    parser.add_argument('--display', default=100, type=int)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--output', default='output', help='The output dir')

    parser.add_argument('--infer_root', default=None)
    parser.add_argument('--dme_weight_aug', default=1.0, type=float)

    return parser.parse_args()

class MultiTaskClsDataSet(torch.utils.data.Dataset):
    def __init__(self, root, config, crop_size, scale_size, baseline=False):
        super(MultiTaskClsDataSet, self).__init__()
        self.root = root
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

    def __getitem__(self, item):
        return self.transform(
            Image.open(os.path.join(self.root, str(self.images_list[item][0]) + '_' + str(self.scale_size) + '.png'))), \
               self.images_list[item][1], self.images_list[item][2], self.images_list[item][3]

    def __len__(self):
        return len(self.images_list)

class MultiTaskClsValDataSet(torch.utils.data.Dataset):
    def __init__(self, root, config, crop_size, scale_size, baseline=False):
        super(MultiTaskClsValDataSet, self).__init__()
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
        self.transform = transforms.Compose([
            transforms.Scale(self.scale_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values, std=std_values),
        ])

    def __getitem__(self, item):
        return self.transform(Image.open(os.path.join(self.root, str(self.images_list[item][0])+'_'+str(self.scale_size)+'.png'))), self.images_list[item][1], self.images_list[item][2], self.images_list[item][3]

    def __len__(self):
        return len(self.images_list)

class MultiTaskClsInferenceDataSet(torch.utils.data.Dataset):
    def __init__(self, root, crop_size, scale_size):
        super(MultiTaskClsInferenceDataSet, self).__init__()
        self.root = root
        self.images_list = glob(os.path.join(root, '*.jpg'))
        self.crop_size = crop_size
        self.scale_size = scale_size
        with open('info.json', 'r') as fp:
            info = json.load(fp)
        mean_values = torch.from_numpy(np.array(info['mean'], dtype=np.float32) / 255)
        std_values = torch.from_numpy(np.array(info['std'], dtype=np.float32) / 255)
        self.transform = transforms.Compose([
            transforms.Scale(self.scale_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values, std=std_values),
        ])

    def __getitem__(self, item):
        return self.transform(self._scale_image(self.images_list[item], self.scale_size)), os.path.basename(self.images_list[item])

    def __len__(self):
        return len(self.images_list)

    def _scale_image(self, imagepath, scale_size):
        # image = Image.open(os.path.join(root, '{}.jpg'.format(index)))
        image = Image.open(imagepath)
        w, h = image.size
        tw, th = (min(w, h), min(w, h))
        image = image.crop((w // 2 - tw // 2, h // 2 - th // 2, w // 2 + tw // 2, h // 2 + th // 2))
        w, h = image.size
        tw, th = (scale_size, scale_size)
        ratio = tw / w
        assert ratio == th / h
        if ratio < 1:
            image = image.resize((tw, th), Image.ANTIALIAS)
        elif ratio > 1:
            image = image.resize((tw, th), Image.CUBIC)
        w, h = image.size
        tw, th = (min(w, h), min(w, h))
        image = image.crop((w // 2 - tw // 2, h // 2 - th // 2, w // 2 + tw // 2, h // 2 + th // 2))
        w, h = image.size
        tw, th = (scale_size, scale_size)
        ratio = tw / w
        assert ratio == th / h
        if ratio < 1:
            image = image.resize((tw, th), Image.ANTIALIAS)
        elif ratio > 1:
            image = image.resize((tw, th), Image.CUBIC)
        return image

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

class multi_task_model(nn.Module):
    def __init__(self, name, inmap, multi_classes, weights=None, scratch=False, outbincls=True):
        super(multi_task_model, self).__init__()
        self.name = name
        self.weights = weights
        self.inmap = inmap
        self.multi_classes = multi_classes
        self.cls0 = None
        self.cls1 = None
        self.cls2 = None
        self.concatenate = None
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
        # if len(multi_classes) == 3:
        #     self.cls2 = nn.Linear(self.planes, multi_classes[2])
        self.cls2 = nn.Linear(self.planes, 2)
        self.concatenate = nn.Linear(9, 2)

        initialize_cls_weights(self.cls0)
        initialize_cls_weights(self.cls1)
        if weights:
            self.load_state_dict(torch.load(weights))

    def forward(self, x):
        feature = self.base(x)
        # when 'inplace=True', some errors occur!!!!!!!!!!!!!!!!!!!!!!
        out = F.relu(feature, inplace=False)
        out = F.avg_pool2d(out, kernel_size=self.featmap).view(feature.size(0), -1)
        out1 = self.cls0(out)
        out2 = self.cls1(out)
        # out3 = self.cls2(out)
        con = torch.cat((out1, out2), dim=1)
        out3 = self.concatenate(con)
        if self.outbincls:
            return out1, out2, out3
        else:
            return out1, out2

def dataset_test():
    print('welcome to dataset test!')

    args = parse_args()

    ds = MultiTaskClsValDataSet(args.root, args.traincsv, args.crop, args.size)

    train_dataloader = DataLoader(MultiTaskClsValDataSet(args.root, args.traincsv, args.crop, args.size), batch_size=args.batch,
                                  shuffle=False, num_workers=args.workers, pin_memory=False)

    for index, (image, dr_level, dme_level) in enumerate(train_dataloader):
        print('dr_level: ', dr_level)
        print('dme_level: ', dme_level)

    print('bye!')

def model_test():
    print('welcome to model test!')
    args = parse_args()
    train_dataloader = DataLoader(MultiTaskClsValDataSet(args.root, args.traincsv, args.crop, args.size),
                                  batch_size=args.batch,
                                  shuffle=False, num_workers=args.workers, pin_memory=False)
    model = multi_task_model(args.model, inmap=args.crop, multi_classes=[5,4])
    optimizer = optim.SGD([{'params': model.base.parameters()},
                       {'params': model.cls0.parameters()},
                       {'params': model.cls1.parameters()}], lr=args.lr, momentum=args.mom, weight_decay=args.wd,
                      nesterov=True)
    model = nn.DataParallel(model).cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = FocalLoss().cuda()
    for epoch in range(3):
        for index, (image, dr_level, dme_level) in enumerate(train_dataloader):
            o_dr, o_dme = model(Variable(image.cuda()))
            o_dr_loss = criterion(o_dr, Variable(dr_level.cuda()))
            o_dme_loss = criterion(o_dme, Variable(dme_level.cuda()))
            loss = 0.5 * o_dr_loss + 0.5 * o_dme_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss: %.4f'%loss.data[0])

def train(train_data_loader, model, criterion, optimizer, epoch, display):
    model.train()
    tot_pred_dr = np.array([], dtype=int)
    tot_label_dr = np.array([], dtype=int)
    tot_pred_dme = np.array([], dtype=int)
    tot_label_dme = np.array([], dtype=int)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracy = AverageMeter()
    losses_dr = AverageMeter()
    losses_dme = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    logger = []
    for index, (image, label_dr, label_dme) in enumerate(train_data_loader):
        data_time.update(time.time()-end)
        o_dr, o_dme = model(Variable(image.cuda()))
        loss_dr = criterion(o_dr, Variable(label_dr.cuda()))
        loss_dme = criterion(o_dme, Variable(label_dme.cuda()))
        loss = 0.5 * loss_dr + 0.5 * loss_dme
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)
        _,pred_dr = torch.max(o_dr, 1)
        _,pred_dme = torch.max(o_dme, 1)
        pred_dr = pred_dr.cpu().data.numpy().squeeze()
        label_dr = label_dr.numpy().squeeze()
        pred_dme = pred_dme.cpu().data.numpy().squeeze()
        label_dme = label_dme.numpy().squeeze()

        tot_pred_dr = np.append(tot_pred_dr, pred_dr)
        tot_label_dr = np.append(tot_label_dr, label_dr)
        tot_pred_dme = np.append(tot_pred_dme, pred_dme)
        tot_label_dme = np.append(tot_label_dme, label_dme)

        #precision
        losses_dr.update(loss_dr.data[0], len(image))
        losses_dme.update(loss_dme.data[0], len(image))
        losses.update(loss.data[0], len(image))


        if index % display == 0:
            dr_accuracy = np.equal(tot_pred_dr, tot_label_dr).sum()/len(tot_pred_dr)
            dme_accuracy = np.equal(tot_pred_dme, tot_label_dme).sum()/len(tot_pred_dme)
            dr_kappa = quadratic_weighted_kappa(tot_label_dr, tot_pred_dr)
            dme_kappa = quadratic_weighted_kappa(tot_label_dme, tot_pred_dme)
            print_info = 'Epoch: [{epoch}][{iter}/{tot}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.avg:.3f}\t ' \
                         'Loss {loss.avg:.4f}\t' \
                         'DR_Loss {dr_loss.avg:.4f}\t' \
                         'DME_Loss {dme_loss.avg:.4f}\t' \
                         'DR_Kappa {dr_kappa:.4f}\t' \
                         'DR_Accuracy {dr_acc:.4f}\t' \
                         'DME_Kappa {dme_kappa:.4f}\t' \
                         'DME_Accuracy {dme_acc:.4f}\t'.format(epoch=epoch, iter=index, tot=len(train_data_loader),
                                                               batch_time=batch_time,
                                                               data_time=data_time,
                                                               loss=losses,
                                                               dr_loss=losses_dr,
                                                               dme_loss=losses_dme,
                                                               dr_acc=dr_accuracy,
                                                               dme_acc=dme_accuracy,
                                                               dr_kappa=dr_kappa,
                                                               dme_kappa=dme_kappa
                                                               )
            print(print_info)
            logger.append(print_info)
    return logger

def train_dr(train_data_loader, model, criterion, optimizer, epoch, display):
    model.train()
    tot_pred_dr = np.array([], dtype=int)
    tot_label_dr = np.array([], dtype=int)
    tot_pred_dme = np.array([], dtype=int)
    tot_label_dme = np.array([], dtype=int)
    tot_pred_bin = np.array([], dtype=int)
    tot_label_bin = np.array([], dtype=int)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracy = AverageMeter()
    losses_dr = AverageMeter()
    losses_dme = AverageMeter()
    losses_bin = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()
    logger = []
    for index, (image, label_dr, label_dme, label_bin) in enumerate(train_data_loader):
        data_time.update(time.time()-end)
        o_dr, o_dme, o_bin = model(Variable(image.cuda()))
        loss_dr = criterion(o_dr, Variable(label_dr.cuda()))
        loss_dme = criterion(o_dme, Variable(label_dme.cuda()))
        loss_bin = criterion(o_bin, Variable(label_bin.cuda()))
        loss = loss_dr
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)
        _,pred_dr = torch.max(o_dr, 1)
        _,pred_dme = torch.max(o_dme, 1)
        _,pred_bin = torch.max(o_bin, 1)
        pred_dr = pred_dr.cpu().data.numpy().squeeze()
        label_dr = label_dr.numpy().squeeze()
        pred_dme = pred_dme.cpu().data.numpy().squeeze()
        label_dme = label_dme.numpy().squeeze()
        pred_bin = pred_bin.cpu().data.numpy().squeeze()
        label_bin = label_bin.numpy().squeeze()

        tot_pred_dr = np.append(tot_pred_dr, pred_dr)
        tot_label_dr = np.append(tot_label_dr, label_dr)
        tot_pred_dme = np.append(tot_pred_dme, pred_dme)
        tot_label_dme = np.append(tot_label_dme, label_dme)
        tot_pred_bin = np.append(tot_pred_bin, pred_bin)
        tot_label_bin = np.append(tot_label_bin, label_bin)


        #precision
        losses_dr.update(loss_dr.data[0], len(image))
        losses_dme.update(loss_dme.data[0], len(image))
        losses_bin.update(loss_bin.data[0], len(image))
        losses.update(loss.data[0], len(image))

        accuracy.update(np.equal(pred_bin, label_bin).sum() / len(label_bin), len(label_bin))


        if index % display == 0:
            dr_accuracy = np.equal(tot_pred_dr, tot_label_dr).sum()/len(tot_pred_dr)
            dme_accuracy = np.equal(tot_pred_dme, tot_label_dme).sum()/len(tot_pred_dme)
            dr_kappa = quadratic_weighted_kappa(tot_label_dr, tot_pred_dr)
            dme_kappa = quadratic_weighted_kappa(tot_label_dme, tot_pred_dme)
            print_info = 'Epoch: [{epoch}][{iter}/{tot}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.avg:.3f}\t ' \
                         'Loss {loss.avg:.4f}\t' \
                         'DR_Loss {dr_loss.avg:.4f}\t' \
                         'DME_Loss {dme_loss.avg:.4f}\t' \
                         'DR_Kappa {dr_kappa:.4f}\t' \
                         'DR_Accuracy {dr_acc:.4f}\t' \
                         'DME_Kappa {dme_kappa:.4f}\t' \
                         'DME_Accuracy {dme_acc:.4f}\t' \
                         'To_Treat_Accuracy {accuracy.avg:.4f}\t'.format(epoch=epoch, iter=index, tot=len(train_data_loader),
                                                               batch_time=batch_time,
                                                               data_time=data_time,
                                                               loss=losses,
                                                               dr_loss=losses_dr,
                                                               dme_loss=losses_dme,
                                                               dr_acc=dr_accuracy,
                                                               dme_acc=dme_accuracy,
                                                               dr_kappa=dr_kappa,
                                                               dme_kappa=dme_kappa,
                                                               accuracy=accuracy
                                                               )
            print(print_info)
            logger.append(print_info)

    return logger

def train_dr_and_dme(train_data_loader, model, criterion, optimizer, epoch, display):
    model.train()
    tot_pred_dr = np.array([], dtype=int)
    tot_label_dr = np.array([], dtype=int)
    tot_pred_dme = np.array([], dtype=int)
    tot_label_dme = np.array([], dtype=int)
    tot_pred_bin = np.array([], dtype=int)
    tot_label_bin = np.array([], dtype=int)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracy = AverageMeter()
    losses_dr = AverageMeter()
    losses_dme = AverageMeter()
    losses_bin = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()
    logger = []
    for index, (image, label_dr, label_dme, label_bin) in enumerate(train_data_loader):
        data_time.update(time.time()-end)
        o_dr, o_dme, o_bin = model(Variable(image.cuda()))
        loss_dr = criterion(o_dr, Variable(label_dr.cuda()))
        loss_dme = criterion(o_dme, Variable(label_dme.cuda()))
        loss_bin = criterion(o_bin, Variable(label_bin.cuda()))
        loss = loss_dr + loss_dme
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)
        _,pred_dr = torch.max(o_dr, 1)
        _,pred_dme = torch.max(o_dme, 1)
        _,pred_bin = torch.max(o_bin, 1)
        pred_dr = pred_dr.cpu().data.numpy().squeeze()
        label_dr = label_dr.numpy().squeeze()
        pred_dme = pred_dme.cpu().data.numpy().squeeze()
        label_dme = label_dme.numpy().squeeze()
        pred_bin = pred_bin.cpu().data.numpy().squeeze()
        label_bin = label_bin.numpy().squeeze()

        tot_pred_dr = np.append(tot_pred_dr, pred_dr)
        tot_label_dr = np.append(tot_label_dr, label_dr)
        tot_pred_dme = np.append(tot_pred_dme, pred_dme)
        tot_label_dme = np.append(tot_label_dme, label_dme)
        tot_pred_bin = np.append(tot_pred_bin, pred_bin)
        tot_label_bin = np.append(tot_label_bin, label_bin)


        #precision
        losses_dr.update(loss_dr.data[0], len(image))
        losses_dme.update(loss_dme.data[0], len(image))
        losses_bin.update(loss_bin.data[0], len(image))
        losses.update(loss.data[0], len(image))

        accuracy.update(np.equal(pred_bin, label_bin).sum() / len(label_bin), len(label_bin))


        if index % display == 0:
            dr_accuracy = np.equal(tot_pred_dr, tot_label_dr).sum()/len(tot_pred_dr)
            dme_accuracy = np.equal(tot_pred_dme, tot_label_dme).sum()/len(tot_pred_dme)
            dr_kappa = quadratic_weighted_kappa(tot_label_dr, tot_pred_dr)
            dme_kappa = quadratic_weighted_kappa(tot_label_dme, tot_pred_dme)
            print_info = 'Epoch: [{epoch}][{iter}/{tot}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.avg:.3f}\t ' \
                         'Loss {loss.avg:.4f}\t' \
                         'DR_Loss {dr_loss.avg:.4f}\t' \
                         'DME_Loss {dme_loss.avg:.4f}\t' \
                         'DR_Kappa {dr_kappa:.4f}\t' \
                         'DR_Accuracy {dr_acc:.4f}\t' \
                         'DME_Kappa {dme_kappa:.4f}\t' \
                         'DME_Accuracy {dme_acc:.4f}\t' \
                         'To_Treat_Accuracy {accuracy.avg:.4f}\t'.format(epoch=epoch, iter=index, tot=len(train_data_loader),
                                                               batch_time=batch_time,
                                                               data_time=data_time,
                                                               loss=losses,
                                                               dr_loss=losses_dr,
                                                               dme_loss=losses_dme,
                                                               dr_acc=dr_accuracy,
                                                               dme_acc=dme_accuracy,
                                                               dr_kappa=dr_kappa,
                                                               dme_kappa=dme_kappa,
                                                               accuracy=accuracy
                                                               )
            print(print_info)
            logger.append(print_info)

    return logger

def train_dme(train_data_loader, model, criterion, optimizer, epoch, display):
    model.train()
    tot_pred_dr = np.array([], dtype=int)
    tot_label_dr = np.array([], dtype=int)
    tot_pred_dme = np.array([], dtype=int)
    tot_label_dme = np.array([], dtype=int)
    tot_pred_bin = np.array([], dtype=int)
    tot_label_bin = np.array([], dtype=int)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracy = AverageMeter()
    losses_dr = AverageMeter()
    losses_dme = AverageMeter()
    losses_bin = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()
    logger = []
    for index, (image, label_dr, label_dme, label_bin) in enumerate(train_data_loader):
        data_time.update(time.time()-end)
        o_dr, o_dme, o_bin = model(Variable(image.cuda()))
        loss_dr = criterion(o_dr, Variable(label_dr.cuda()))
        loss_dme = criterion(o_dme, Variable(label_dme.cuda()))
        loss_bin = criterion(o_bin, Variable(label_bin.cuda()))
        loss = loss_dme
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)
        _,pred_dr = torch.max(o_dr, 1)
        _,pred_dme = torch.max(o_dme, 1)
        _,pred_bin = torch.max(o_bin, 1)
        pred_dr = pred_dr.cpu().data.numpy().squeeze()
        label_dr = label_dr.numpy().squeeze()
        pred_dme = pred_dme.cpu().data.numpy().squeeze()
        label_dme = label_dme.numpy().squeeze()
        pred_bin = pred_bin.cpu().data.numpy().squeeze()
        label_bin = label_bin.numpy().squeeze()

        tot_pred_dr = np.append(tot_pred_dr, pred_dr)
        tot_label_dr = np.append(tot_label_dr, label_dr)
        tot_pred_dme = np.append(tot_pred_dme, pred_dme)
        tot_label_dme = np.append(tot_label_dme, label_dme)
        tot_pred_bin = np.append(tot_pred_bin, pred_bin)
        tot_label_bin = np.append(tot_label_bin, label_bin)


        #precision
        losses_dr.update(loss_dr.data[0], len(image))
        losses_dme.update(loss_dme.data[0], len(image))
        losses_bin.update(loss_bin.data[0], len(image))
        losses.update(loss.data[0], len(image))

        accuracy.update(np.equal(pred_bin, label_bin).sum() / len(label_bin), len(label_bin))


        if index % display == 0:
            dr_accuracy = np.equal(tot_pred_dr, tot_label_dr).sum()/len(tot_pred_dr)
            dme_accuracy = np.equal(tot_pred_dme, tot_label_dme).sum()/len(tot_pred_dme)
            dr_kappa = quadratic_weighted_kappa(tot_label_dr, tot_pred_dr)
            dme_kappa = quadratic_weighted_kappa(tot_label_dme, tot_pred_dme)
            print_info = 'Epoch: [{epoch}][{iter}/{tot}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.avg:.3f}\t ' \
                         'Loss {loss.avg:.4f}\t' \
                         'DR_Loss {dr_loss.avg:.4f}\t' \
                         'DME_Loss {dme_loss.avg:.4f}\t' \
                         'DR_Kappa {dr_kappa:.4f}\t' \
                         'DR_Accuracy {dr_acc:.4f}\t' \
                         'DME_Kappa {dme_kappa:.4f}\t' \
                         'DME_Accuracy {dme_acc:.4f}\t' \
                         'To_Treat_Accuracy {accuracy.avg:.4f}\t'.format(epoch=epoch, iter=index, tot=len(train_data_loader),
                                                               batch_time=batch_time,
                                                               data_time=data_time,
                                                               loss=losses,
                                                               dr_loss=losses_dr,
                                                               dme_loss=losses_dme,
                                                               dr_acc=dr_accuracy,
                                                               dme_acc=dme_accuracy,
                                                               dr_kappa=dr_kappa,
                                                               dme_kappa=dme_kappa,
                                                               accuracy=accuracy
                                                               )
            print(print_info)
            logger.append(print_info)

    return logger

def train_bin(train_data_loader, model, criterion, optimizer, epoch, display, dme_weight_aug_ratio=1):
    model.train()
    tot_pred_dr = np.array([], dtype=int)
    tot_label_dr = np.array([], dtype=int)
    tot_pred_dme = np.array([], dtype=int)
    tot_label_dme = np.array([], dtype=int)
    tot_pred_bin = np.array([], dtype=int)
    tot_label_bin = np.array([], dtype=int)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracy = AverageMeter()
    losses_dr = AverageMeter()
    losses_dme = AverageMeter()
    losses_bin = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()
    logger = []
    for index, (image, label_dr, label_dme, label_bin) in enumerate(train_data_loader):
        data_time.update(time.time()-end)
        o_dr, o_dme, o_bin = model(Variable(image.cuda()))
        loss_dr = criterion(o_dr, Variable(label_dr.cuda()))
        # loss_dme = criterion(o_dme, Variable(label_dme.cuda()))

        dme_weight_aug = torch.ones(4)

        dme_weight_aug[0] = 1
        dme_weight_aug[1] = dme_weight_aug_ratio
        dme_weight_aug[2] = dme_weight_aug_ratio
        dme_weight_aug[3] = dme_weight_aug_ratio

        criterion_dme = nn.CrossEntropyLoss(dme_weight_aug.cuda()).cuda()

        loss_dme = criterion_dme(o_dme, Variable(label_dme.cuda()))

        loss_bin = criterion(o_bin, Variable(label_bin.cuda()))
        loss = loss_bin + loss_dr + loss_dme
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)

        _, pred_dr = torch.max(o_dr, 1)
        _, pred_dme = torch.max(o_dme, 1)
        _, pred_bin = torch.max(o_bin, 1)
        pred_dr = pred_dr.cpu().data.numpy().squeeze()
        label_dr = label_dr.numpy().squeeze()
        pred_dme = pred_dme.cpu().data.numpy().squeeze()
        label_dme = label_dme.numpy().squeeze()
        pred_bin = pred_bin.cpu().data.numpy().squeeze()
        label_bin = label_bin.numpy().squeeze()

        tot_pred_dr = np.append(tot_pred_dr, pred_dr)
        tot_label_dr = np.append(tot_label_dr, label_dr)
        tot_pred_dme = np.append(tot_pred_dme, pred_dme)
        tot_label_dme = np.append(tot_label_dme, label_dme)
        tot_pred_bin = np.append(tot_pred_bin, pred_bin)
        tot_label_bin = np.append(tot_label_bin, label_bin)


        #precision
        losses_dr.update(loss_dr.data[0], len(image))
        losses_dme.update(loss_dme.data[0], len(image))
        losses_bin.update(loss_bin.data[0], len(image))
        losses.update(loss.data[0], len(image))

        accuracy.update(np.equal(pred_bin, label_bin).sum() / len(label_bin), len(label_bin))


        if index % display == 0:
            dr_accuracy = np.equal(tot_pred_dr, tot_label_dr).sum()/len(tot_pred_dr)
            dme_accuracy = np.equal(tot_pred_dme, tot_label_dme).sum()/len(tot_pred_dme)
            dr_kappa = quadratic_weighted_kappa(tot_label_dr, tot_pred_dr)
            dme_kappa = quadratic_weighted_kappa(tot_label_dme, tot_pred_dme)
            print_info = 'Epoch: [{epoch}][{iter}/{tot}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.avg:.3f}\t ' \
                         'Loss {loss.avg:.4f}\t' \
                         'DR_Loss {dr_loss.avg:.4f}\t' \
                         'DME_Loss {dme_loss.avg:.4f}\t' \
                         'DR_Kappa {dr_kappa:.4f}\t' \
                         'DR_Accuracy {dr_acc:.4f}\t' \
                         'DME_Kappa {dme_kappa:.4f}\t' \
                         'DME_Accuracy {dme_acc:.4f}\t' \
                         'To_Treat_Accuracy {accuracy.avg:.4f}\t'.format(epoch=epoch, iter=index, tot=len(train_data_loader),
                                                               batch_time=batch_time,
                                                               data_time=data_time,
                                                               loss=losses,
                                                               dr_loss=losses_dr,
                                                               dme_loss=losses_dme,
                                                               dr_acc=dr_accuracy,
                                                               dme_acc=dme_accuracy,
                                                               dr_kappa=dr_kappa,
                                                               dme_kappa=dme_kappa,
                                                               accuracy=accuracy
                                                               )
            print(print_info)
            logger.append(print_info)

    return logger

def eval(eval_data_loader, model, criterion):
    model.eval()
    tot_pred_dr = np.array([], dtype=int)
    tot_label_dr = np.array([], dtype=int)
    tot_pred_dme = np.array([], dtype=int)
    tot_label_dme = np.array([], dtype=int)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracy = AverageMeter()
    losses_dr = AverageMeter()
    losses_dme = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    logger = []
    for index, (image, label_dr, label_dme) in enumerate(eval_data_loader):
        data_time.update(time.time()-end)
        o_dr, o_dme = model(Variable(image.cuda()))
        loss_dr = criterion(o_dr, Variable(label_dr.cuda()))
        loss_dme = criterion(o_dme, Variable(label_dme.cuda()))
        loss = 0.5 * loss_dr + 0.5 * loss_dme
        batch_time.update(time.time()-end)
        _,pred_dr = torch.max(o_dr, 1)
        _,pred_dme = torch.max(o_dme, 1)
        pred_dr = pred_dr.cpu().data.numpy().squeeze()
        label_dr = label_dr.numpy().squeeze()
        pred_dme = pred_dme.cpu().data.numpy().squeeze()
        label_dme = label_dme.numpy().squeeze()

        tot_pred_dr = np.append(tot_pred_dr, pred_dr)
        tot_label_dr = np.append(tot_label_dr, label_dr)
        tot_pred_dme = np.append(tot_pred_dme, pred_dme)
        tot_label_dme = np.append(tot_label_dme, label_dme)

        #precision
        losses_dr.update(loss_dr.data[0], len(image))
        losses_dme.update(loss_dme.data[0], len(image))
        losses.update(loss.data[0], len(image))

        dr_accuracy = np.equal(tot_pred_dr, tot_label_dr).sum() / len(tot_pred_dr)
        dme_accuracy = np.equal(tot_pred_dme, tot_label_dme).sum() / len(tot_pred_dme)
        dr_kappa = quadratic_weighted_kappa(tot_label_dr, tot_pred_dr)
        dme_kappa = quadratic_weighted_kappa(tot_label_dme, tot_pred_dme)
        print_info = 'Eval: [{iter}/{tot}]\t' \
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                     'Data {data_time.avg:.3f}\t ' \
                     'Loss {loss.avg:.4f}\t' \
                     'DR_Loss {dr_loss.avg:.4f}\t' \
                     'DME_Loss {dme_loss.avg:.4f}\t' \
                     'DR_Kappa {dr_kappa:.4f}\t' \
                     'DR_Accuracy {dr_acc:.4f}\t' \
                     'DME_Kappa {dme_kappa:.4f}\t' \
                     'DME_Accuracy {dme_acc:.4f}\t'.format(iter=index, tot=len(eval_data_loader),
                                                           batch_time=batch_time,
                                                           data_time=data_time,
                                                           loss=losses,
                                                           dr_loss=losses_dr,
                                                           dme_loss=losses_dme,
                                                           dr_acc=dr_accuracy,
                                                           dme_acc=dme_accuracy,
                                                           dr_kappa=dr_kappa,
                                                           dme_kappa=dme_kappa
                                                           )
        print(print_info)
        logger.append(print_info)

    return logger, dr_kappa, dme_kappa, tot_pred_dr, tot_label_dr, tot_pred_dme, tot_label_dme

'''
sensitivity:

tp/(tp+fn)

specificity:

tn/(tn+fp)
'''
def calc_sensitivity_specificity(pred, label):
    assert len(pred) == len(label)
    tp_cnt = 0
    tn_cnt = 0
    fp_cnt = 0
    fn_cnt = 0
    # print(pred)
    # print(label)
    for i in range(0, len(pred)):
        if pred[i] == 1:
            if label[i] == 1:
                tp_cnt += 1
            else:
                fp_cnt += 1
        else:
            if label[i] == 1:
                fn_cnt += 1
            else:
                tn_cnt += 1

    sensitivity = tp_cnt/(tp_cnt+fn_cnt) if (tp_cnt+fn_cnt)>0 else 0
    specificity = tn_cnt/(tn_cnt+fp_cnt) if (tn_cnt+fp_cnt)>0 else 0
    f1 = 2 * tp_cnt / (2 * tp_cnt + fp_cnt + fn_cnt) if (2 * tp_cnt + fp_cnt + fn_cnt) > 0 else 0


    return sensitivity, specificity, f1

def eval_bin(eval_data_loader, model, criterion):
    model.eval()
    tot_pred_dr = np.array([], dtype=int)
    tot_label_dr = np.array([], dtype=int)
    tot_pred_dme = np.array([], dtype=int)
    tot_label_dme = np.array([], dtype=int)
    tot_pred_bin = np.array([], dtype=int)
    tot_label_bin = np.array([], dtype=int)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracy = AverageMeter()
    losses_dr = AverageMeter()
    losses_dme = AverageMeter()
    losses_bin = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    logger = []
    for index, (image, label_dr, label_dme, label_bin) in enumerate(eval_data_loader):
        data_time.update(time.time()-end)
        o_dr, o_dme, o_bin = model(Variable(image.cuda()))
        loss_dr = criterion(o_dr, Variable(label_dr.cuda()))
        loss_dme = criterion(o_dme, Variable(label_dme.cuda()))
        loss_bin = criterion(o_bin, Variable(label_bin.cuda()))
        loss = 0.5 * loss_dr + 0.5 * loss_dme
        batch_time.update(time.time()-end)
        _,pred_dr = torch.max(o_dr, 1)
        _,pred_dme = torch.max(o_dme, 1)
        _, pred_bin = torch.max(o_bin, 1)
        pred_dr = pred_dr.cpu().data.numpy().squeeze()
        label_dr = label_dr.numpy().squeeze()
        pred_dme = pred_dme.cpu().data.numpy().squeeze()
        label_dme = label_dme.numpy().squeeze()
        pred_bin = pred_bin.cpu().data.numpy().squeeze()
        label_bin = label_bin.numpy().squeeze()

        tot_pred_dr = np.append(tot_pred_dr, pred_dr)
        tot_label_dr = np.append(tot_label_dr, label_dr)
        tot_pred_dme = np.append(tot_pred_dme, pred_dme)
        tot_label_dme = np.append(tot_label_dme, label_dme)
        tot_pred_bin = np.append(tot_pred_bin, pred_bin)
        tot_label_bin = np.append(tot_label_bin, label_bin)

        #precision
        losses_dr.update(loss_dr.data[0], len(image))
        losses_dme.update(loss_dme.data[0], len(image))
        losses_bin.update(loss_bin.data[0], len(image))
        losses.update(loss.data[0], len(image))

        dr_accuracy = np.equal(tot_pred_dr, tot_label_dr).sum() / len(tot_pred_dr)
        dme_accuracy = np.equal(tot_pred_dme, tot_label_dme).sum() / len(tot_pred_dme)
        dr_kappa = quadratic_weighted_kappa(tot_label_dr, tot_pred_dr)
        dme_kappa = quadratic_weighted_kappa(tot_label_dme, tot_pred_dme)

        accuracy.update(np.equal(pred_bin, label_bin).sum() / len(label_bin), len(label_bin))

        print_info = 'Eval: [{iter}/{tot}]\t' \
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                     'Data {data_time.avg:.3f}\t ' \
                     'Loss {loss.avg:.4f}\t' \
                     'DR_Loss {dr_loss.avg:.4f}\t' \
                     'DME_Loss {dme_loss.avg:.4f}\t' \
                     'To_Treat_Loss {bin_loss.avg:.4f}\t' \
                     'DR_Kappa {dr_kappa:.4f}\t' \
                     'DR_Accuracy {dr_acc:.4f}\t' \
                     'DME_Kappa {dme_kappa:.4f}\t' \
                     'DME_Accuracy {dme_acc:.4f}\t' \
                     'To_Treat_Accuray {accuracy.avg:.4f}\t'.format(iter=index, tot=len(eval_data_loader),
                                                           batch_time=batch_time,
                                                           data_time=data_time,
                                                           loss=losses,
                                                           dr_loss=losses_dr,
                                                           dme_loss=losses_dme,
                                                           dr_acc=dr_accuracy,
                                                           dme_acc=dme_accuracy,
                                                           dr_kappa=dr_kappa,
                                                           dme_kappa=dme_kappa,
                                                           bin_loss=losses_bin,
                                                                accuracy=accuracy,
                                                           )
        print(print_info)
        logger.append(print_info)

    sensitivity, specificity, f1 = calc_sensitivity_specificity(tot_pred_bin, tot_label_bin)
    print_info1 = '\nbinary cls accuracy: {0:.4f}\tsensitivity: {1:.4f}\t specificity: {2:.4f}\tf1 score: {3:.4f}'.format(accuracy.avg, sensitivity, specificity, f1)
    logger.append(print_info1)
    print(print_info1)

    tot_pred_dr_bin = [0 if x <=1 else 1 for x in tot_pred_dr]
    # tot_label_dr_bin = [0 if x <= 1 else 1 for x in tot_label_dr]
    tot_label_dr_bin = tot_label_bin
    dr_bin_accuray = np.equal(tot_pred_dr_bin, tot_label_dr_bin).sum() / len(tot_label_dr_bin)
    dr_s1, dr_s2, dr_f1 = calc_sensitivity_specificity(tot_pred_dr_bin, tot_label_dr_bin)

    log_dr_bin_cls = '\n[DR binary cls]: acc: {acc:.4f}\tsensitivity: {s1:.4f}\tspecificity: {s2:.4f}\tf1 score: {f1:.4f}'.format(
        acc = dr_bin_accuray,
        s1 = dr_s1,
        s2 = dr_s2, f1 = dr_f1
    )

    print(log_dr_bin_cls)

    logger.append(log_dr_bin_cls)


    tot_pred_dme_bin = [0 if x < 1 else 1 for x in tot_pred_dme]
    # tot_label_dme_bin = [0 if x < 1 else 1 for x in tot_label_dme]
    tot_label_dme_bin = tot_label_bin
    dme_bin_accuray = np.equal(tot_pred_dme_bin, tot_label_dme_bin).sum() / len(tot_label_dme_bin)
    calc_sensitivity_specificity(tot_pred_dme_bin, tot_label_dme_bin)
    dme_s1, dme_s2, dme_f1 = calc_sensitivity_specificity(tot_pred_dme_bin, tot_label_dme_bin)
    log_dme_bin_cls = '\n[DME binary cls]: acc: {acc:.4f}\tsensitivity: {s1:.4f}\tspecificity: {s2:.4f}\tf1 score: {f1:.4f}'.format(
        acc=dme_bin_accuray,
        s1=dme_s1,
        s2=dme_s2, f1 = dme_f1
    )
    print(log_dme_bin_cls)
    logger.append(log_dme_bin_cls)

    # tot_pred_mix_bin = [0 if (x < 1 or y <=1) else 1 for x in tot_pred_dme for y in tot_pred_dr]
    cnt = 0
    for i in range(len(tot_pred_dr)):
        numdr = tot_pred_dr[i]
        numdme = tot_pred_dme[i]
        if numdr < 2:
            if numdme > 0:
                cnt += 1

    print('dr < 2 and dme >0 count is: {}'.format(cnt))

    cnt = 0
    for i in range(len(tot_pred_dr)):
        if tot_pred_dme[i] == 0:
            cnt += 1
    print('pred dme 0 count is: {}'.format(cnt))

    cnt = 0
    for i in range(len(tot_pred_dr)):
        if tot_label_dme[i] == 0:
            cnt += 1
    print('label dme 0 count is: {}'.format(cnt))

    tot_pred_mix_bin = [0 if (tot_pred_dme[i] == 0 and tot_pred_dr[i] <= 1) else 1 for i in range(len(tot_pred_dme))]
    # tot_label_mix_bin = [0 if (x < 1 or y <=1) else 1 for x in tot_label_dme for y in tot_label_dr]
    # tot_label_mix_bin = [0 if (tot_label_dme[i] < 1 and tot_label_dr[i] <= 1) else 1 for i in range(len(tot_label_dme))]
    tot_label_mix_bin = tot_label_bin
    mix_bin_accuray = np.equal(tot_pred_mix_bin, tot_label_mix_bin).sum() / len(tot_label_mix_bin)
    calc_sensitivity_specificity(tot_pred_mix_bin, tot_label_mix_bin)
    mix_s1, mix_s2, mix_f1 = calc_sensitivity_specificity(tot_pred_mix_bin, tot_label_mix_bin)
    log_mix_bin_cls = '\n[MIX binary cls]: acc: {acc:.4f}\tsensitivity: {s1:.4f}\tspecificity: {s2:.4f}\tf1 score: {f1:.4f}'.format(
        acc=mix_bin_accuray,
        s1=mix_s1,
        s2=mix_s2, f1 = mix_f1
    )
    print(log_mix_bin_cls)
    logger.append(log_mix_bin_cls)

    return logger, dr_kappa, dme_kappa, tot_pred_dr, tot_label_dr, tot_pred_dme, tot_label_dme, accuracy.avg

def infer_bin(eval_data_loader, model, refer_root):
    model.eval()
    tot_pred_dr = np.array([], dtype=int)
    tot_label_dr = np.array([], dtype=int)
    tot_pred_dme = np.array([], dtype=int)
    tot_label_dme = np.array([], dtype=int)
    tot_pred_bin = np.array([], dtype=int)
    tot_label_bin = np.array([], dtype=int)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    logger = []
    imagenames = []
    for index, (image, imagename) in enumerate(eval_data_loader):
        for name in imagename:
            imagenames.append(name)
        data_time.update(time.time()-end)
        o_dr, o_dme, o_bin = model(Variable(image.cuda()))
        batch_time.update(time.time()-end)
        _,pred_dr = torch.max(o_dr, 1)
        _,pred_dme = torch.max(o_dme, 1)
        _, pred_bin = torch.max(o_bin, 1)
        pred_dr = pred_dr.cpu().data.numpy().squeeze()
        pred_dme = pred_dme.cpu().data.numpy().squeeze()
        pred_bin = pred_bin.cpu().data.numpy().squeeze()

        tot_pred_dr = np.append(tot_pred_dr, pred_dr)
        tot_pred_dme = np.append(tot_pred_dme, pred_dme)
        tot_pred_bin = np.append(tot_pred_bin, pred_bin)

        data = np.column_stack((imagenames, tot_pred_dr, tot_pred_dme, tot_pred_bin))
        data_df = pd.DataFrame(data, columns=['image', 'dr_level', 'dme_level', 'referable'])

        data_csv = os.path.join(refer_root, 'result.csv')

        data_df.to_csv(data_csv)

        print_info = 'Eval: [{iter}/{tot}]\t'.format(iter=index, tot=len(eval_data_loader))
        print(print_info)

    print('====>reference end!')

    return logger



def train_test():
    print('welcome to train test!')
    args = parse_args()
    train_dataloader = DataLoader(MultiTaskClsValDataSet(args.root, args.traincsv, args.crop, args.size),
                                  batch_size=args.batch,
                                  shuffle=False, num_workers=args.workers, pin_memory=False)
    model = multi_task_model(args.model, inmap=args.crop, multi_classes=[5, 4])
    optimizer = optim.SGD([{'params': model.base.parameters()},
                           {'params': model.cls0.parameters()},
                           {'params': model.cls1.parameters()}], lr=args.lr, momentum=args.mom, weight_decay=args.wd,
                          nesterov=True)
    # model = nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    for epoch in range(3):
        logger = train(train_dataloader, nn.DataParallel(model).cuda(), criterion, optimizer, epoch, 10)

def main():
    print('===> Parsing options')
    opt = parse_args()
    print(opt)
    cudnn.benchmark = True
    torch.manual_seed(opt.seed)
    if not os.path.isdir(opt.output):
        os.makedirs(opt.output)
    time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    output_dir = os.path.join(opt.output,
                              opt.dataset + '_multi_task_cls_' + opt.phase + '_' + time_stamp + '_' + opt.model + '_' + opt.exp)
    if not os.path.exists(output_dir):
        print('====> Creating ', output_dir)
        os.makedirs(output_dir)

    print('====> Building model:')
    model = multi_task_model(opt.model, inmap=opt.crop, multi_classes=[5, 4], weights=opt.weight)
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = FocalLoss(5).cuda()

    if opt.phase == 'train':
        print('====> Training model:')
        dataset_train = DataLoader(MultiTaskClsDataSet(opt.root, opt.traincsv, opt.crop, opt.size),
                                  batch_size=opt.batch,
                                  shuffle=True, num_workers=opt.workers, pin_memory=True)
        dataset_val = DataLoader(MultiTaskClsValDataSet(opt.root, opt.valcsv, opt.crop, opt.size),
                                  batch_size=opt.batch,
                                  shuffle=False, num_workers=opt.workers, pin_memory=False)
        kp_dr_best = 0
        kp_dme_best = 0
        bin_acc_best = 0
        for epoch in range(opt.epoch):
            if epoch < opt.fix:
                lr = opt.lr
            else:
                lr = opt.lr * (0.1 ** (epoch // opt.step))
            optimizer = optim.SGD([{'params': model.base.parameters()},
                                   {'params': model.cls0.parameters()},
                                   {'params': model.cls1.parameters()}], lr=opt.lr, momentum=opt.mom,
                                  weight_decay=opt.wd,
                                  nesterov=True)
            optimizer_dr_only = optim.SGD([{'params': model.base.parameters()},
                                           {'params': model.cls0.parameters()}], lr=opt.lr, momentum=opt.mom,
                                          weight_decay=opt.wd,
                                          nesterov=True)
            optimizer_dme_only = optim.SGD([{'params': model.base.parameters()},
                                            {'params': model.cls1.parameters()}], lr=opt.lr, momentum=opt.mom,
                                           weight_decay=opt.wd,
                                           nesterov=True)
            optimizer_bin_only = optim.SGD([{'params': model.base.parameters()},
                                            {'params': model.cls2.parameters()}], lr=opt.lr, momentum=opt.mom,
                                           weight_decay=opt.wd,
                                           nesterov=True)
            optimizer_dr_ft = optim.SGD([{'params': model.cls0.parameters()}], lr=opt.lr, momentum=opt.mom,
                                        weight_decay=opt.wd,
                                        nesterov=True)
            optimizer_dme_ft = optim.SGD([{'params': model.cls1.parameters()}], lr=opt.lr, momentum=opt.mom,
                                         weight_decay=opt.wd,
                                         nesterov=True)
            optimizer_bin_ft = optim.SGD([{'params': model.cls2.parameters()}], lr=opt.lr, momentum=opt.mom,
                                         weight_decay=opt.wd,
                                         nesterov=True)
            optimizer_dme_ft = optim.SGD([{'params': model.cls1.parameters()}], lr=opt.lr, momentum=opt.mom,
                                         weight_decay=opt.wd,
                                         nesterov=True)
            optimizer_bin_ft = optim.SGD([{'params': model.cls2.parameters()}], lr=opt.lr, momentum=opt.mom,
                                         weight_decay=opt.wd,
                                         nesterov=True)
            optimizer_bin_con = optim.SGD([{'params': model.base.parameters()},
                                   {'params': model.cls0.parameters()},
                                   {'params': model.cls1.parameters()}, {'params': model.concatenate.parameters()},], lr=opt.lr, momentum=opt.mom,
                                  weight_decay=opt.wd,
                                  nesterov=True)
            # logger = train(dataset_train, nn.DataParallel(model).cuda(), criterion, optimizer, epoch, opt.display)

            # if epoch % 500 < 200:
            #     logger_dme_ft = train_dme(dataset_train, nn.DataParallel(model).cuda(), criterion, optimizer_dme_ft, epoch, opt.display)
            #     logger_bin_ft = train_bin(dataset_train, nn.DataParallel(model).cuda(), criterion, optimizer_bin_ft,
            #                               epoch, opt.display)
            #     logger = logger_bin_ft
            # else:
            #     logger_dr = train_dr(dataset_train, nn.DataParallel(model).cuda(), criterion, optimizer_dr_only, epoch, opt.display)
            #     logger = logger_dr

            # logger = train_dr_and_dme(dataset_train, nn.DataParallel(model).cuda(), criterion, optimizer, epoch,
            #                           opt.display)

            logger = train_bin(dataset_train, nn.DataParallel(model).cuda(), criterion, optimizer_bin_con, epoch,
                                      opt.display, opt.dme_weight_aug)

            # logger_val, kp_dr, kp_dme, _,_,_,_ = eval(dataset_val, nn.DataParallel(model).cuda(), criterion)
            logger_val, kp_dr, kp_dme, _, _, _, _, acc = eval_bin(dataset_val, nn.DataParallel(model).cuda(), criterion)

            if kp_dr > kp_dr_best:
                print('\ncurrent best dr kappa is: {}\n'.format(kp_dr))
                kp_dr_best = kp_dr
                torch.save(model.cpu().state_dict(), os.path.join(output_dir,
                                                                  opt.dataset + '_multi_task_cls_dr_' + opt.model + '_%03d' % epoch + '_best.pth'))
                print('====> Save model: {}'.format(
                    os.path.join(output_dir, opt.dataset + '_multi_task_cls_dr_' + opt.model + '_%03d' % epoch + '_best.pth')))

            if kp_dme > kp_dme_best:
                print('\ncurrent best dme kappa is: {}\n'.format(kp_dme))
                kp_dme_best = kp_dme
                torch.save(model.cpu().state_dict(), os.path.join(output_dir,
                                                                  opt.dataset + '_multi_task_cls_dme_' + opt.model + '_%03d' % epoch + '_best.pth'))
                print('====> Save model: {}'.format(
                    os.path.join(output_dir, opt.dataset + '_multi_task_cls_dme_' + opt.model + '_%03d' % epoch + '_best.pth')))

            if acc > bin_acc_best:
                print('\ncurrent best concatenation binary cls accuracy is: {}\n'.format(acc))
                bin_acc_best = acc
                torch.save(model.cpu().state_dict(), os.path.join(output_dir,
                                                                  opt.dataset + '_multi_task_cls_concatenation_binary_' + opt.model + '_%03d' % epoch + '_best.pth'))
                print('====> Save model: {}'.format(
                    os.path.join(output_dir,
                                 opt.dataset + '_multi_task_cls_concatenation_binary_' + opt.model + '_%03d' % epoch + '_best.pth')))

            if not os.path.isfile(os.path.join(output_dir, 'train.log')):
                with open(os.path.join(output_dir, 'train.log'), 'w') as fp:
                    fp.write(str(opt)+'\n\n')
            with open(os.path.join(output_dir, 'train.log'), 'a') as fp:
                fp.write('\n' + '\n'.join(logger))
                fp.write('\n' + '\n'.join(logger_val))
    elif opt.phase == 'test':
        if opt.weight:
            print('====> Evaluating model')
            dataset_test = DataLoader(MultiTaskClsValDataSet(opt.root, opt.testcsv, opt.crop, opt.size),
                                  batch_size=opt.batch,
                                  shuffle=False, num_workers=opt.workers, pin_memory=False)
            logger_val, kp_dr, kp_dme, pred_dr, label_dr, pred_dme, label_dme, _ = eval_bin(dataset_test, nn.DataParallel(model).cuda(), criterion)
            print('===> DR Kappa: %.4f' % kp_dr)
            print('===> Confusion Matrix:')
            dr_confusion_matrix = str(confusion_matrix(label_dr, pred_dr))
            print(dr_confusion_matrix)
            print('\n\n')
            print('===> DME Kappa: %.4f' % kp_dme)
            print('===> Confusion Matrix:')
            dme_confusion_matrix = str(confusion_matrix(label_dme, pred_dme))
            print(dme_confusion_matrix)
            print('\n\n')

            with open(os.path.join(output_dir, 'test.log'), 'w') as fp:
                fp.write(str(opt) + '\n')
                fp.write('\n' + '\n'.join(logger_val))
                fp.write('\n====> DR Kappa: %.4f' % kp_dr)
                fp.write('\n')
                fp.write(dr_confusion_matrix)
                fp.write('\n')
                fp.write('\n====> DME Kappa: %.4f' % kp_dme)
                fp.write('\n')
                fp.write(dme_confusion_matrix)
            np.save(os.path.join(output_dir, 'results_dr.npy'), pred_dr, label_dr, confusion_matrix(label_dr, pred_dr))
            np.save(os.path.join(output_dir, 'results_dme.npy'), pred_dme, label_dme, confusion_matrix(label_dme, pred_dme))
        else:
            raise Exception('No weights found!')
    elif opt.phase == 'infer':
        if opt.weight:
            print('====> Evaluating model')
            dataset_refer = DataLoader(MultiTaskClsInferenceDataSet(opt.infer_root, opt.crop, opt.size),
                                  batch_size=opt.batch,
                                  shuffle=False, num_workers=opt.workers, pin_memory=False)
            logger_val = infer_bin(dataset_refer, nn.DataParallel(model).cuda(), opt.infer_root)
        else:
            raise Exception('No weights found!')
    else:
        raise Exception('No phase found')


if __name__ == '__main__':
    # dataset_test()
    # model_test()
    # train_test()
    main()