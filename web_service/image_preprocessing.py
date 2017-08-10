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
        return torch.nn.DataParallel(model).cuda()

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





def main():
    # image = Image.open('/Users/zhangweidong03/Code/dr/DiabeticRetinopathy_solution/image_processing_test/C0025307.jpg')
    # cropped_image = get_input_image(image)
    # cropped_image.show()

    classifier = DrImageClassifier('rsn34', 'kaggle.pth', [2])

    imagepath = 'test.jpeg'
    image = Image.open(imagepath)

    idx, prop = classifier.classifyImage(image)

    print('DR level is: {}'.format(idx))
    print('propobality distribution is: {}'.format(prop))


if __name__  == '__main__':
    main()
