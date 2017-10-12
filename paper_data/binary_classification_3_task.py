import json
import torch.utils.data
import numpy as np
from PIL import Image
import sys
sys.path.append('../')
from utils import TenCrop, HorizontalFlip, Affine, ColorJitter, Lighting, PILColorJitter
from utils import AverageMeter, quadratic_weighted_kappa
import torchvision.transforms as transforms
import pandas as pd

import argparse
import os

from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import math

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch.backends.cudnn as cudnn

import time


def parse_args():
    parser = argparse.ArgumentParser(description='binary classification')

    parser.add_argument('--root', required=True)
    parser.add_argument('--traincsv', default=None)
    parser.add_argument('--valcsv', default=None)
    parser.add_argument('--testcsv', default=None)

    parser.add_argument('--exp', default='binary_cls', help='The name of experiment')
    parser.add_argument('--dataset', default='kaggle', choices=['kaggle'], help='The dataset to use')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--model', default='googlenet', choices=['googlenet', 'rsn18', 'rsn34', 'rsn50', 'rsn101', 'rsn150', 'dsn121', 'dsn161', 'dsn169', 'dsn201',])
    parser.add_argument('--batch', default=8, type=int, help='The batch size of training')
    parser.add_argument('--crop', default=448, type=int, help='The crop size of input')
    parser.add_argument('--size', default=512, type=int, choices=[128,256,512,1024], help='The scale size of input')
    parser.add_argument('--weight', default=None, help='The path of pretrained model')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--mom', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--fix', default=100, type=int)
    parser.add_argument('--step', default=100, type=int)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--display', default=10, type=int, help='The frequency of print log')
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--output', default='output', help='The output dir')

    return parser.parse_args()

opt = parse_args()

print(opt)

class BinClsDataSet(torch.utils.data.Dataset):
    def __init__(self, root, config, crop_size, scale_size, baseline=False, gcn=True):
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
        if gcn:
            if baseline:
                self.transform = transforms.Compose([
                    transforms.RandomCrop(crop_size),
                    transforms.Scale(299),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean_values, std=std_values),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomCrop(crop_size),
                    transforms.Scale(299),
                    transforms.RandomHorizontalFlip(),
                    PILColorJitter(),
                    transforms.ToTensor(),
                    Lighting(alphastd=0.01, eigval=eigen_values, eigvec=eigen_values),
                    transforms.Normalize(mean=mean_values, std=std_values),
                ])
        else:
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
        return self.transform(Image.open(os.path.join(self.root, self.images_list[item][0]+'_'+str(self.scale_size)+'.png'))), 0 if self.images_list[item][1] < 2 else 1, 0 if self.images_list[item][1] < 2 else 1, 0 if self.images_list[item][2] < 1 else 1

    def __len__(self):
        return len(self.images_list)

class BinClsDataSetVal(torch.utils.data.Dataset):
    def __init__(self, root, config, crop_size, scale_size, baseline=False, gcn=True):
        super(BinClsDataSetVal, self).__init__()
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
        if gcn:
            self.transform = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.Scale(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_values, std=std_values),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(crop_size),
                # transforms.Scale(scale_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_values, std=std_values),
            ])

    def __getitem__(self, item):
        return self.transform(Image.open(os.path.join(self.root, self.images_list[item][0]+'_'+str(self.scale_size)+'.png'))), 0 if self.images_list[item][1] < 2 else 1, 0 if self.images_list[item][1] < 2 else 1, 0 if self.images_list[item][2] < 1 else 1

    def __len__(self):
        return len(self.images_list)

def test_dataset():
    dataset = DataLoader(BinClsDataSet(opt.root, opt.labelscsv, 224, 512), batch_size=16)
    for index, (images, labels) in enumerate(dataset):
        torchvision.utils.save_image(images, './test.jpeg')
        print(labels)

def initial_cls_weights(cls):
    for m in cls.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class TestCls(nn.Module):
    def __init__(self, model, weights=None):
        super(TestCls, self).__init__()
        self.model = model

        self.base_model = nn.Sequential(
            *list(model.children())[0:3],
            nn.MaxPool2d(3,2),
            *list(model.children())[3:5],
            nn.MaxPool2d(3, 2),
            *list(model.children())[5:13],
            *list(model.children())[14:-1],
            nn.AvgPool2d(kernel_size=8),
            nn.Dropout(),
        )

        self.base_model0 = nn.Sequential(*list(model.children())[0:3])
        self.base_model1 = nn.Sequential(*list(model.children())[3:5])
        self.base_model2 = nn.Sequential(*list(model.children())[5:13])
        self.base_model3 = nn.Sequential(*list(model.children())[14:-1])

        self.aux = list(model.children())[13]
        self.fc = nn.Linear(2048, 2)
        self.fc_bin = nn.Linear(2048, 2)
        self.fc_dr = nn.Linear(2048, 2)
        self.fc_dme = nn.Linear(2048, 2)
        self.cls = nn.Sequential(nn.Conv2d(2048, 5, kernel_size=1, stride=1, padding=0, bias=True))
        initial_cls_weights(self.cls)
        if weights:
            self.load_state_dict(torch.load(weights))
    def forward(self, x):
        x = self.base_model0(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.base_model1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.base_model2(x)
        x = self.base_model3(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x_bin = self.fc_bin(x)
        x_dr = self.fc_dr(x)
        x_dme = self.fc_dme(x)
        return x_bin, x_dr, x_dme


class multi_channel_model(nn.Module):
    def __init__(self, name, inmap, multi_classes, weights=None, scratch=False):
        super(multi_channel_model, self).__init__()
        self.name = name
        self.weights = weights
        self.inmap = inmap
        self.multi_classes = multi_classes
        self.cls0 = None
        self.cls1 = None
        self.cls2 = None
        self.featmap = inmap // 32
        self.planes = 2048
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

        self.cls = nn.Linear(self.planes, multi_classes)
        self.cls_dr = nn.Linear(self.planes, multi_classes)
        self.cls_dme = nn.Linear(self.planes, multi_classes)

        initial_cls_weights(self.cls)
        initial_cls_weights(self.cls_dr)
        initial_cls_weights(self.cls_dme)

        if weights:
            self.load_state_dict(torch.load(weights))

    def forward(self, x):
        feature = self.base(x)
        # when 'inplace=True', some errors occur!!!!!!!!!!!!!!!!!!!!!!
        out = F.relu(feature, inplace=False)
        out = F.avg_pool2d(out, kernel_size=self.featmap).view(feature.size(0), -1)
        out_bin = self.cls(out)
        out_dr = self.cls_dr(out)
        out_dme = self.cls_dme(out)
        return out_bin, out_dr, out_dme




def cls_train_3_task(train_data_loader, model, criterion, optimizer, epoch, display, ft_dme=False):
    model.train()

    tot_pred_bin = np.array([], dtype=int)
    tot_label_bin = np.array([], dtype=int)
    tot_pred_dr = np.array([], dtype=int)
    tot_label_dr = np.array([], dtype=int)
    tot_pred_dme = np.array([], dtype=int)
    tot_label_dme = np.array([], dtype=int)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_bin = AverageMeter()
    accuracy_dr = AverageMeter()
    accuracy_dme = AverageMeter()
    end = time.time()
    logger = []
    for num_iter, (images, labels_bin, labels_dr, labels_dme) in enumerate(train_data_loader):
        data_time.update(time.time()-end)
        o_bin, o_dr, o_dme = model(Variable(images.cuda()))
        loss_bin = criterion(o_bin, Variable(labels_bin.cuda()))
        loss_dr = criterion(o_dr, Variable(labels_dr.cuda()))
        loss_dme = criterion(o_dme, Variable(labels_dme.cuda()))
        optimizer.zero_grad()
        # loss = loss_bin + loss_dr + loss_dme
        if not ft_dme:
            loss = loss_bin + loss_dr
        else:
            loss = loss_dme
        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)
        _,pred_bin = torch.max(o_bin, 1)
        _, pred_dr = torch.max(o_dr, 1)
        _, pred_dme = torch.max(o_dme, 1)

        pred_bin = pred_bin.cpu().data.numpy().squeeze()
        labels_bin = labels_bin.numpy().squeeze()
        pred_dr = pred_dr.cpu().data.numpy().squeeze()
        labels_dr = labels_dr.numpy().squeeze()
        pred_dme = pred_dme.cpu().data.numpy().squeeze()
        labels_dme = labels_dme.numpy().squeeze()

        tot_pred_bin = np.append(tot_pred_bin, pred_bin)
        tot_label_bin = np.append(tot_label_bin, labels_bin)
        tot_pred_dr = np.append(tot_pred_dr, pred_dr)
        tot_label_dr = np.append(tot_label_dr, labels_dr)
        tot_pred_dme = np.append(tot_pred_dme, pred_dme)
        tot_label_dme = np.append(tot_label_dme, labels_dme)

        losses.update(loss.data[0], len(images))
        accuracy_bin.update(np.equal(pred_bin, labels_bin).sum()/len(labels_bin), len(labels_bin))
        accuracy_dr.update(np.equal(pred_dr, labels_dr).sum() / len(labels_dr), len(labels_dr))
        accuracy_dme.update(np.equal(pred_dme, labels_dme).sum() / len(labels_dme), len(labels_dme))

        end = time.time()
        if num_iter % display == 0:
            correct_bin = np.equal(tot_pred_bin, tot_label_bin).sum()/len(tot_pred_bin)
            correct_dr = np.equal(tot_pred_bin, tot_label_bin).sum() / len(tot_pred_bin)
            correct_dme = np.equal(tot_pred_bin, tot_label_bin).sum() / len(tot_pred_bin)

            print_info = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\t'\
                'Data {data_time.avg:.3f}\t''Loss {loss.avg:.4f}\tDR Graded Accuray {accuracy_bin.avg:.4f}' \
                         '\tDR Referable Accuray {accuracy_dr.avg:.4f}' \
                         '\tDME Referable Accuray {accuracy_dme.avg:.4f}'.format(
                epoch, num_iter, len(train_data_loader),batch_time=batch_time, data_time=data_time,
                loss=losses, accuracy_bin=accuracy_bin, accuracy_dr=accuracy_dr, accuracy_dme=accuracy_dme
            )
            print(print_info)
            logger.append(print_info)
    return logger


def cls_train(train_data_loader, model, criterion, optimizer, epoch, display):
    model.train()
    tot_pred = np.array([], dtype=int)
    tot_label = np.array([], dtype=int)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()
    logger = []
    for num_iter, (images, labels) in enumerate(train_data_loader):
        data_time.update(time.time()-end)
        output = model(Variable(images.cuda()))
        loss = criterion(output, Variable(labels.cuda()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)
        _,pred = torch.max(output, 1)
        pred = pred.cpu().data.numpy().squeeze()
        labels = labels.numpy().squeeze()
        tot_pred = np.append(tot_pred, pred)
        tot_label = np.append(tot_label, labels)
        losses.update(loss.data[0], len(images))
        accuracy.update(np.equal(pred, labels).sum()/len(labels), len(labels))
        end = time.time()
        if num_iter % display == 0:
            correct = np.equal(tot_pred, tot_label).sum()/len(tot_pred)
            print_info = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\t'\
                'Data {data_time.avg:.3f}\t''Loss {loss.avg:.4f}\tAccuray {accuracy.avg:.4f}'.format(
                epoch, num_iter, len(train_data_loader),batch_time=batch_time, data_time=data_time,
                loss=losses, accuracy=accuracy
            )
            print(print_info)
            logger.append(print_info)
    return logger



'''
sensitivity:

tp/(tp+fn)

specificity:

tn/(tn+fp)

f1 score:
2tp/(2tp+fp+fn)
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
    f1 = 2*tp_cnt/(2*tp_cnt+fp_cnt+fn_cnt) if (2*tp_cnt+fp_cnt+fn_cnt) > 0 else 0

    return sensitivity, specificity, f1


def cls_eval_3_task(eval_data_loader, model, criterion, display):
    model.eval()

    tot_pred_bin = np.array([], dtype=int)
    tot_label_bin = np.array([], dtype=int)
    tot_pred_dr = np.array([], dtype=int)
    tot_label_dr = np.array([], dtype=int)
    tot_pred_dme = np.array([], dtype=int)
    tot_label_dme = np.array([], dtype=int)

    tot_prob_bin = np.array([], dtype=float)
    tot_prob_dr = np.array([], dtype=float)
    tot_prob_dme = np.array([], dtype=float)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_bin = AverageMeter()
    accuracy_dr = AverageMeter()
    accuracy_dme = AverageMeter()
    end = time.time()
    logger = []
    for num_iter, (image, labels_bin, labels_dr, labels_dme) in enumerate(eval_data_loader):
        data_time.update(time.time()-end)
        o_bin, o_dr, o_dme = model(Variable(image.cuda()))

        loss_bin = criterion(o_bin, Variable(labels_bin.cuda()))
        loss_dr = criterion(o_dr, Variable(labels_dr.cuda()))
        loss_dme = criterion(o_dme, Variable(labels_dme.cuda()))
        loss = loss_bin + loss_dr + loss_dme
        losses.update(loss.data[0], len(image))

        _, pred_bin = torch.max(o_bin, 1)
        _, pred_dr = torch.max(o_dr, 1)
        _, pred_dme = torch.max(o_dme, 1)

        m = torch.nn.Softmax()
        prop = m(o_bin).data
        res_prop = prop.cpu().numpy()
        res_prop = res_prop[:, 1]
        tot_prob_bin = np.append(tot_prob_bin, res_prop)
        prop = m(o_dr).data
        res_prop = prop.cpu().numpy()
        res_prop = res_prop[:, 1]
        tot_prob_dr = np.append(tot_prob_dr, res_prop)
        prop = m(o_dme).data
        res_prop = prop.cpu().numpy()
        res_prop = res_prop[:, 1]
        tot_prob_dme = np.append(tot_prob_dme, res_prop)

        pred_bin = pred_bin.cpu().data.numpy().squeeze()
        labels_bin = labels_bin.numpy().squeeze()
        pred_dr = pred_dr.cpu().data.numpy().squeeze()
        labels_dr = labels_dr.numpy().squeeze()
        pred_dme = pred_dme.cpu().data.numpy().squeeze()
        labels_dme = labels_dme.numpy().squeeze()

        tot_pred_bin = np.append(tot_pred_bin, pred_bin)
        tot_label_bin = np.append(tot_label_bin, labels_bin)
        tot_pred_dr = np.append(tot_pred_dr, pred_dr)
        tot_label_dr = np.append(tot_label_dr, labels_dr)
        tot_pred_dme = np.append(tot_pred_dme, pred_dme)
        tot_label_dme = np.append(tot_label_dme, labels_dme)

        batch_time.update(time.time()-end)

        accuracy_bin.update(np.equal(pred_bin, labels_bin).sum() / len(labels_bin), len(labels_bin))
        accuracy_dr.update(np.equal(pred_dr, labels_dr).sum() / len(labels_dr), len(labels_dr))
        accuracy_dme.update(np.equal(pred_dme, labels_dme).sum() / len(labels_dme), len(labels_dme))

        end = time.time()
        print_info = 'Eval: [{0}/{1}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\t' \
                     'Data {data_time.avg:.3f}\t''Loss {loss.avg:.4f}\tAccuray {accuracy_bin.avg:.4f}' \
                         '\tDR Referable Accuray {accuracy_dr.avg:.4f}' \
                         '\tDME Referable Accuray {accuracy_dme.avg:.4f}'.format(
            num_iter, len(eval_data_loader), batch_time=batch_time, data_time=data_time,
            loss=losses, accuracy_bin=accuracy_bin, accuracy_dr=accuracy_dr, accuracy_dme=accuracy_dme
        )
        logger.append(print_info)
        print(print_info)

    from sklearn.metrics import roc_curve, auc
    # bin
    fpr, tpr, _ = roc_curve(tot_label_bin, tot_prob_bin)
    roc_auc = auc(fpr, tpr)
    print('fpr: {}'.format(fpr))
    print('tpr: {}'.format(tpr))
    print('ground truth label: {}'.format(tot_label_bin))
    print('probability label: {}'.format(tot_prob_bin))
    print('roc_auc: {}'.format(roc_auc))

    import matplotlib.pyplot as plt

    fpr = fpr * 100
    tpr = tpr * 100

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 100], [0, 100], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 100.0])
    plt.ylim([0.0, 105])
    plt.xlabel('Specificity,%')
    plt.ylabel('Sensitivity,%')
    plt.title('google dr bin')
    plt.legend(loc="lower right")
    plt.show()

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 100], [0, 100], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 30.0])
    plt.ylim([70.0, 105])
    plt.xlabel('Specificity,%')
    plt.ylabel('Sensitivity,%')
    plt.title('google dr bin')
    plt.legend(loc="lower right")
    plt.show()

    # dr bin
    fpr, tpr, _ = roc_curve(tot_label_dr, tot_prob_dr)
    roc_auc = auc(fpr, tpr)
    print('fpr: {}'.format(fpr))
    print('tpr: {}'.format(tpr))
    print('ground truth label: {}'.format(tot_label_dr))
    print('probability label: {}'.format(tot_prob_dr))
    print('roc_auc: {}'.format(roc_auc))

    import matplotlib.pyplot as plt

    fpr = fpr * 100
    tpr = tpr * 100

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 100], [0, 100], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 100.0])
    plt.ylim([0.0, 105])
    plt.xlabel('Specificity,%')
    plt.ylabel('Sensitivity,%')
    plt.title('google dr referable')
    plt.legend(loc="lower right")
    plt.show()

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 100], [0, 100], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 30.0])
    plt.ylim([70.0, 105])
    plt.xlabel('Specificity,%')
    plt.ylabel('Sensitivity,%')
    plt.title('google dr referable')
    plt.legend(loc="lower right")
    plt.show()

    # dme bin
    fpr, tpr, _ = roc_curve(tot_label_dr, tot_prob_dr)
    roc_auc = auc(fpr, tpr)
    print('fpr: {}'.format(fpr))
    print('tpr: {}'.format(tpr))
    print('ground truth label: {}'.format(tot_label_dr))
    print('probability label: {}'.format(tot_prob_dr))
    print('roc_auc: {}'.format(roc_auc))

    import matplotlib.pyplot as plt

    fpr = fpr * 100
    tpr = tpr * 100

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 100], [0, 100], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 100.0])
    plt.ylim([0.0, 105])
    plt.xlabel('Specificity,%')
    plt.ylabel('Sensitivity,%')
    plt.title('google dme referable')
    plt.legend(loc="lower right")
    plt.show()

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 100], [0, 100], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 30.0])
    plt.ylim([70.0, 105])
    plt.xlabel('Specificity,%')
    plt.ylabel('Sensitivity,%')
    plt.title('google dme referable')
    plt.legend(loc="lower right")
    plt.show()

    sensitivity_bin, specificity, f1 = calc_sensitivity_specificity(tot_pred_bin, tot_label_bin)
    print_info1 = '\n[DR Graded:]\taccuracy:{0:.4f}\tsensitivity: {1:.4f}\t specificity: {2:.4f}\tf1 score: {3:.4f}\n'.format(accuracy_bin.avg, sensitivity_bin, specificity, f1)

    sensitivity_dr, specificity, f1 = calc_sensitivity_specificity(tot_pred_dr, tot_label_dr)
    print_info1_dr = '\n[DR Referable:]\taccuracy:{0:.4f}\tsensitivity: {1:.4f}\t specificity: {2:.4f}\tf1 score: {3:.4f}\n'.format(
        accuracy_dr.avg, sensitivity_dr, specificity, f1)

    sensitivity_dme, specificity, f1 = calc_sensitivity_specificity(tot_pred_dme, tot_label_dme)
    print_info1_dme = '\n[DME Referable:]\taccuracy:{0:.4f}\tsensitivity: {1:.4f}\t specificity: {2:.4f}\tf1 score: {3:.4f}\n'.format(
        accuracy_dme.avg, sensitivity_dme, specificity, f1)

    logger.append(print_info1)
    logger.append(print_info1_dr)
    logger.append(print_info1_dme)

    print(print_info1)
    print(print_info1_dr)
    print(print_info1_dme)

    # return (accuracy_bin.avg+accuracy_dr.avg+accuracy_dme.avg)/3, logger
    return (accuracy_bin.avg + accuracy_dr.avg) / 2, logger, sensitivity_bin, sensitivity_dr, sensitivity_dme, accuracy_dme.avg

def cls_eval(eval_data_loader, model, criterion, display):
    model.eval()
    tot_pred = np.array([], dtype=int)
    tot_label = np.array([], dtype=int)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()
    logger = []
    for num_iter, (image, label) in enumerate(eval_data_loader):
        data_time.update(time.time()-end)
        output = model(Variable(image.cuda()))
        loss = criterion(output, Variable(label.cuda()))
        _,pred = torch.max(output, 1)
        pred = pred.cpu().data.numpy().squeeze()
        label = label.numpy().squeeze()
        losses.update(loss.data[0], len(image))
        batch_time.update(time.time()-end)

        tot_pred = np.append(tot_pred, pred)
        tot_label = np.append(tot_label, label)

        accuracy.update(np.equal(pred, label).sum()/len(label), len(label))
        end = time.time()
        print_info = 'Eval: [{0}/{1}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\t' \
                     'Data {data_time.avg:.3f}\t''Loss {loss.avg:.4f}\tAccuray {accuracy.avg:.4f}'.format(
            num_iter, len(eval_data_loader), batch_time=batch_time, data_time=data_time,
            loss=losses, accuracy=accuracy
        )
        logger.append(print_info)
        print(print_info)

    sensitivity, specificity, f1 = calc_sensitivity_specificity(tot_pred, tot_label)
    print_info1 = '\naccuracy:{0:.4f}\tsensitivity: {1:.4f}\t specificity: {2:.4f}\tf1 score: {3:.4f}\n'.format(accuracy.avg, sensitivity, specificity, f1)
    logger.append(print_info1)
    print(print_info1)

    return accuracy.avg, logger

def main():
    print('===> Parsing options')
    opt = parse_args()
    print(opt)
    cudnn.benchmark = True
    torch.manual_seed(opt.seed)
    if not torch.cuda.is_available():
        raise Exception('No GPU found!')
    if not os.path.isdir(opt.output):
        os.makedirs(opt.output)
    time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    output_dir = os.path.join(opt.output, opt.dataset+'_binary_cls_'+opt.phase+'_'+time_stamp+'_'+opt.model+'_'+opt.exp)
    if not os.path.exists(output_dir):
        print('====> Creating ', output_dir)
        os.makedirs(output_dir)

    print('====> Building model:')

    if opt.model == 'googlenet':
        model = torchvision.models.inception_v3(True)
        model = TestCls(model, opt.weight)
    else:
        model = multi_channel_model(opt.model, inmap=opt.crop, multi_classes=2, weights=opt.weight)

    model_cuda = nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    gcn = True if opt.model == 'googlenet' else False

    if opt.phase == 'train':
        print('====> Training model')

        dataset_train = DataLoader(BinClsDataSet(opt.root, opt.traincsv, opt.crop, opt.size, gcn=gcn), batch_size=opt.batch, num_workers=opt.workers,
                             shuffle=True, pin_memory=True)
        dataset_val = DataLoader(BinClsDataSetVal(opt.root, opt.valcsv, opt.crop, opt.size, gcn=gcn), batch_size=opt.batch,
                             num_workers=opt.workers,
                             shuffle=False, pin_memory=False)
        accuracy_best = 0
        accuracy_dme_best = 0
        s_bin_best = 0
        s_dr_best = 0
        s_dme_best = 0
        for epoch in range(opt.epoch):
            if epoch < opt.fix:
                lr = opt.lr
            else:
                lr = opt.lr * (0.1 ** (epoch // opt.step))

            optimizer = None

            train_dme = False
            if opt.model == 'googlenet':
                optimizer = optim.SGD(
                    [{'params': model.base_model0.parameters()}, {'params': model.base_model1.parameters()},
                     {'params': model.base_model2.parameters()},
                     {'params': model.base_model3.parameters()},
                     # {'params': model.fc.parameters()},
                     {'params': model.fc_bin.parameters()},
                     {'params': model.fc_dr.parameters()},
                     {'params': model.fc_dme.parameters()}],
                    lr=lr,
                    momentum=opt.mom,
                    weight_decay=opt.wd,
                    nesterov=True)
                optimizer_bin = optim.SGD(
                    [{'params': model.base_model0.parameters()}, {'params': model.base_model1.parameters()},
                     {'params': model.base_model2.parameters()},
                     {'params': model.base_model3.parameters()},
                     # {'params': model.fc.parameters()},
                     {'params': model.fc_bin.parameters()},
                     {'params': model.fc_dr.parameters()},
                     # {'params': model.fc_dme.parameters()},
                     ],
                    lr=lr,
                    momentum=opt.mom,
                    weight_decay=opt.wd,
                    nesterov=True)
                optimizer_dme = optim.SGD(
                    [{'params': model.fc_dme.parameters()}],
                    lr=lr,
                    momentum=opt.mom,
                    weight_decay=opt.wd,
                    nesterov=True)
                if epoch >= 0:
                    optimizer = optimizer_dme
                    train_dme = True
                else:
                    optimizer = optimizer_bin
                    train_dme = False
            else:
                optimizer = optim.SGD([{'params': model.base.parameters()},
                                       {'params': model.cls.parameters()}], lr=opt.lr, momentum=opt.mom,
                                      weight_decay=opt.wd,
                                      nesterov=True)

            logger = cls_train_3_task(dataset_train, nn.DataParallel(model).cuda(), criterion, optimizer, epoch, opt.display, train_dme)

            acc, logger_val, s_bin, s_dr, s_dme, acc_dme = cls_eval_3_task(dataset_val, nn.DataParallel(model).cuda(), criterion, opt.display)

            if acc > accuracy_best:
                print('\ncurrent best accuracy is: {}\n'.format(acc))
                accuracy_best = acc
                torch.save(model.cpu().state_dict(), os.path.join(output_dir, opt.dataset+'_binarycls_'+opt.model+'_%03d'%epoch+'_best.pth'))
                print('====> Save model: {}'.format(os.path.join(output_dir, opt.dataset+'_binarycls_'+opt.model+'_%03d'%epoch+'_best.pth')))
            if acc_dme > accuracy_dme_best:
                print('\ncurrent best dme accuracy is: {}\n'.format(acc_dme))
                accuracy_dme_best = acc_dme
                torch.save(model.cpu().state_dict(), os.path.join(output_dir, opt.dataset+'_binarycls_'+opt.model+'_%03d'%epoch+'_dme_best.pth'))
                print('====> Save model: {}'.format(os.path.join(output_dir, opt.dataset+'_binarycls_'+opt.model+'_%03d'%epoch+'_dme_best.pth')))
            if s_bin > s_bin_best:
                print('\ncurrent best bin sensitivity is: {}\n'.format(s_bin))
                s_bin_best = s_bin
                torch.save(model.cpu().state_dict(), os.path.join(output_dir, opt.dataset+'_binarycls_'+opt.model+'_%03d'%epoch+'_bin_sensitivity_best.pth'))
                print('====> Save model: {}'.format(os.path.join(output_dir, opt.dataset+'_binarycls_'+opt.model+'_%03d'%epoch+'_bin_sensitivity_best.pth')))
            if s_dr > s_dr_best:
                print('\ncurrent best dr sensitivity is: {}\n'.format(s_dr))
                s_dr_best = s_dr
                torch.save(model.cpu().state_dict(), os.path.join(output_dir, opt.dataset+'_binarycls_'+opt.model+'_%03d'%epoch+'_dr_sensitivity_best.pth'))
                print('====> Save model: {}'.format(os.path.join(output_dir, opt.dataset+'_binarycls_'+opt.model+'_%03d'%epoch+'_dr_sensitivity_best.pth')))
            if s_dme > s_dme_best:
                print('\ncurrent best dme sensitivity is: {}\n'.format(s_dme))
                s_dme_best = s_dme
                torch.save(model.cpu().state_dict(), os.path.join(output_dir, opt.dataset+'_binarycls_'+opt.model+'_%03d'%epoch+'_dme_sensitivity_best.pth'))
                print('====> Save model: {}'.format(os.path.join(output_dir, opt.dataset+'_binarycls_'+opt.model+'_%03d'%epoch+'_dme_sensitivity_best.pth')))
            if not os.path.isfile(os.path.join(output_dir, 'train.log')):
                with open(os.path.join(output_dir, 'train.log'), 'w') as fp:
                    fp.write(str(opt)+'\n\n')
            with open(os.path.join(output_dir, 'train.log'), 'a') as fp:
                fp.write('\n' + '\n'.join(logger))
                fp.write('\n' + '\n'.join(logger_val))
    elif opt.phase == 'test':
        if opt.weight:
            print('====> Evaluating model')
            dataset_test = DataLoader(BinClsDataSetVal(opt.root, opt.testcsv, opt.size, opt.size, gcn=gcn), batch_size=opt.batch,
                                      num_workers=opt.workers,
                                      shuffle=False, pin_memory=False)
            acc, logger_test, _, _, _, _ = cls_eval_3_task(dataset_test, nn.DataParallel(model).cuda(), criterion, opt.display)
            with open(os.path.join(output_dir, 'test.log'), 'w') as fp:
                fp.write('\n' + '\n'.join(logger_test))
                fp.write('\n' + str(opt) + '\n')
                fp.write('\n====> Accuracy: %.4f' % acc)
            print('\n====> Accuracy: %.4f' % acc)
        else:
            raise Exception('No weights found!')
    else:
        raise Exception('No phase found')

if __name__ == '__main__':
    main()

