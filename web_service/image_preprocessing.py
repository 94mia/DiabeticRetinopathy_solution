from PIL import Image

import sys
sys.path.append('..')

import json

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import drn

import numpy as np

import math

import argparse


__all__ = [
    'DrImageClassifier',
    'get_kaggle_classifier',
    'get_zz_classifier',
    'get_all_classifier',
]


parser = argparse.ArgumentParser()
parser.add_argument('-dev','--devlist', nargs='+', help='<Required> Set flag',
                    type=int, default=[0,1,2,3], required=False)
args = parser.parse_args()
devs = args.devlist
torch.cuda.set_device(devs[0])


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


class cls_model(nn.Module):
	def __init__(self, name, inmap, classes, weights, scratch, cls2reg):
		super(cls_model, self).__init__()
		if name == 'drn26':
			base_model = drn.drn26()
			featmap = inmap // 8
			planes = 512
		elif name == 'drn42':
			base_model = drn.drn42()
			featmap = inmap // 8
			planes = 512
		elif name == 'rsn152':
			base_model = torchvision.models.resnet152()
			featmap = inmap // 32
			planes = 2048
		elif name == 'rsn101':
			base_model = torchvision.models.resnet101()
			featmap = inmap // 32
			planes = 2048
		elif name == 'rsn50':
			base_model = torchvision.models.resnet50()
			featmap = inmap // 32
			planes = 2048
		elif name == 'rsn34':
			base_model = torchvision.models.resnet34()
			featmap = inmap // 32
			planes = 512
		elif name == 'rsn18':
			base_model = torchvision.models.resnet18()
			featmap = inmap // 32
			planes = 512
		if not scratch:
			base_model.load_state_dict(torch.load('pretrained/' + name + '.pth'))
		self.base = nn.Sequential(*list(base_model.children())[:-2])
		if cls2reg:
			cls = nn.Sequential(nn.AvgPool2d(featmap), nn.Conv2d(planes, 1, kernel_size=1, stride=1, padding=0, bias=True))
		else:
		  cls = nn.Sequential(nn.AvgPool2d(featmap), nn.Conv2d(planes, classes, kernel_size=1, stride=1, padding=0, bias=True))
		initialize_cls_weights(cls)
		self.cls = cls
		if weights:
			self.load_state_dict(torch.load(weights))

	def forward(self, x):
		x = self.base(x)
		x = self.cls(x)
		x = x.view(x.size(0), -1)
		return x

def get_input_image(image):
    w,h = image.size
    tw, th = (min(w, h), min(w, h))
    image = image.crop((w // 2 - tw // 2, h // 2 - th // 2, w // 2 + tw // 2, h // 2 + th // 2))
    w,h = image.size
    tw = 512
    th = 512
    ratio = tw/w
    assert ratio == th/h
    image = image.resize((tw, th), Image.ANTIALIAS)
    return image

class DrImageClassifier(object):
    def __init__(self, arch, weights, devs=[0]):
        self.arch = arch
        self.weights = weights
        self.devs =devs
        self.init_crop = get_input_image
        self.rescale_size = 512
        self.crop_size = 512
        self.model_loaded = False
        with open('../data/kaggle/info.json', 'r') as fp:
            info = json.load(fp)
            print(info)
        mean_values = torch.from_numpy(np.array(info['mean'], dtype=np.float32) / 255)
        std_values = torch.from_numpy(np.array(info['std'], dtype=np.float32) / 255)
        self.transform = transforms.Compose([
            transforms.Scale(self.rescale_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values, std=std_values),
        ])

    def load_model(self, arch, weights, devs=[0]):
        model = cls_model(arch, self.crop_size, 5, weights, True, False)
        print('device id is: '.format(devs))
        return torch.nn.DataParallel(model, devs).cuda()

    def image_preprocessed(self, image):
        cropped_image = self.init_crop(image)
        batch_imgs = torch.stack([self.transform(cropped_image)])
        return batch_imgs

    def classifyImage(self, image):
        if not self.model_loaded:
            self.model = self.load_model(self.arch, self.weights, self.devs)
            self.model.eval()
            self.model_loaded = True
        input = self.image_preprocessed(image)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output = self.model(input_var)
        pred = output.data.max(1)[1]
        m = torch.nn.Softmax()
        # prop = m(output).data.max(1)[0]
        prop = m(output).data
        res = 0
        use_cuda = True
        if use_cuda:
            res = pred.cpu().numpy()
            res_prop = prop.cpu().numpy()
        else:
            res = pred.numpy()
            res_prop = prop.numpy()
        return res[0][0], res_prop


import torch.nn.functional as F

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

class DrImageClassifier_ZZ(object):
    def __init__(self, arch, weights, devs=[0]):
        self.arch = arch
        self.weights = weights
        self.devs =devs
        self.init_crop = get_input_image
        self.rescale_size = 512
        self.crop_size = 512
        self.model_loaded = False
        with open('../data/kaggle/info.json', 'r') as fp:
            info = json.load(fp)
            print(info)
        mean_values = torch.from_numpy(np.array(info['mean'], dtype=np.float32) / 255)
        std_values = torch.from_numpy(np.array(info['std'], dtype=np.float32) / 255)
        self.transform = transforms.Compose([
            transforms.Scale(self.rescale_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values, std=std_values),
        ])

    def load_model(self, arch, weights, devs=[0]):
        model = cls_model(arch, self.crop_size, multi_classes=[5, 4], weights=weights)
        print('device id is: '.format(devs))
        return torch.nn.DataParallel(model, devs).cuda()

    def image_preprocessed(self, image):
        cropped_image = self.init_crop(image)
        batch_imgs = torch.stack([self.transform(cropped_image)])
        return batch_imgs

    def classifyImage(self, image):
        if not self.model_loaded:
            self.model = self.load_model(self.arch, self.weights, self.devs)
            self.model.eval()
            self.model_loaded = True
        input = self.image_preprocessed(image)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output,_,_ = self.model(input_var)
        pred = output.data.max(1)[1]
        m = torch.nn.Softmax()
        # prop = m(output).data.max(1)[0]
        prop = m(output).data
        res = 0
        use_cuda = True
        if use_cuda:
            res = pred.cpu().numpy()
            res_prop = prop.cpu().numpy()
        else:
            res = pred.numpy()
            res_prop = prop.numpy()
        return res[0][0], res_prop


def get_kaggle_classifier():
    classifier = DrImageClassifier('rsn34', 'kaggle.pth', args.devlist)
    return classifier

def get_zz_classifier():
    # classifier = DrImageClassifier('rsn34', 'zz.pth', args.devlist)
    classifier = DrImageClassifier_ZZ('rsn34', 'zz.pth', args.devlist)
    return classifier

def get_all_classifier():
    classifier = DrImageClassifier('rsn34', 'all.pth', args.devlist)
    return classifier


def main():
    # image = Image.open('/Users/zhangweidong03/Code/dr/DiabeticRetinopathy_solution/image_processing_test/C0025307.jpg')
    # cropped_image = get_input_image(image)
    # cropped_image.show()

    classifier = DrImageClassifier('rsn34', 'kaggle.pth', args.devlist)

    imagepath = 'test.jpeg'
    image = Image.open(imagepath)

    idx, prop = classifier.classifyImage(image)

    print('DR level is: {}'.format(idx))
    print('propobality distribution is: {}'.format(prop))


if __name__  == '__main__':
    main()
