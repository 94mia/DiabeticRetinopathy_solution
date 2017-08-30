import argparse, json, math, time, os, torch, torchvision.models
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import drn
from utils import quadratic_weighted_kappa, AverageMeter
from data import *

from torchvision.utils import make_grid, save_image

def parse_args():
    parser = argparse.ArgumentParser(description='extract feature maps')

    parser.add_argument('--seed', default=111, type=int, help='Random seed to use')

    return parser.parse_args()



import json
import torch.utils.data
import numpy as np
from PIL import Image
from utils import TenCrop, HorizontalFlip, Affine, ColorJitter, Lighting, PILColorJitter
import torchvision.transforms as transforms


class kaggleClsTrain(torch.utils.data.Dataset):

	def __init__(self, crop_size, scale_size, baseline):
		super(kaggleClsTrain, self).__init__()
		self.image = ['data/tmp/512/' + line.strip() + '_' + str(scale_size) + '.png' for line in open('data/tmp/val_images.txt', 'r')]
		self.label = torch.from_numpy(np.array(np.loadtxt('data/tmp/val_labels.txt'), np.int))
		with open('data/kaggle/info.json', 'r') as fp:
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
				transforms.RandomSizedCrop(crop_size),
				transforms.RandomHorizontalFlip(),
				PILColorJitter(),
				transforms.ToTensor(),
				#ColorJitter(),
				Lighting(alphastd=0.1, eigval=eigen_values, eigvec=eigen_vectors),
				#Affine(rotation_range=180, translation_range=None, shear_range=None, zoom_range=None),
				transforms.Normalize(mean=mean_values, std=std_values),
			])

	def __getitem__(self, index):
		return self.transform(Image.open(self.image[index])), self.label[index]

	def __len__(self):
		if len(self.image) != len(self.label):
			raise Exception("The number of images and labels should be the same.")
		return len(self.label)




def initial_cls_weights(cls):
    for m in cls.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
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
        # if not scratch:
        #     base_model.load_state_dict(torch.load('pretrained/'+name+'.pth'))
        self.base = nn.Sequential(*list(base_model.children())[:-5])
        self.layer3 = list(base_model.children())[-5]
        self.layer4 = list(base_model.children())[-4]
        self.layer5 = list(base_model.children())[-3]
        cls = nn.Sequential(nn.AvgPool2d(featmap), nn.Conv2d(planes, classes, kernel_size=1, stride=1, padding=0, bias=True))
        initial_cls_weights(cls)
        self.cls = cls

    def forward(self, x):
        map = self.base(x)
        y = self.layer3(map)
        y = self.layer4(y)
        y = self.layer5(y)
        y = self.cls(y)
        y = y.view(y.size(0), -1)
        return y, map


# model = cls_model('rsn34', 448, 5, None, None, None)



def cls_train(train_data_loader, model, criterion, optimizer, epoch, display):
    model.train()
    for num_iter, (image, label) in enumerate(train_data_loader):
        final, map = model(Variable(image))
        loss = criterion(final, Variable(label.cuda()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        fmap = map.cpu().data[0][1]
        mean = torch.mean(fmap)
        std = torch.std(fmap)
        trans1 = transforms.Normalize(mean=mean, std=std)
        trans = transforms.ToPILImage()
        fmap.sub_(mean).div_(std)
        pic = fmap.mul(255).byte()
        pil_image = Image.fromarray(pic.numpy(), mode='L')
        if num_iter % 50 == 0:
            # pil_image.show()
            grid = make_grid(map.cpu().data[:,2,:,:].unsqueeze(1), normalize=True)
            ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            im = Image.fromarray(ndarr)
            im.show()
            save_image(map.cpu().data[:,2,:,:].unsqueeze(1), 'test.jpeg', normalize=True)



def main():
    print('====> Parsing options')
    opt = parse_args()
    print(opt)
    cudnn.benchmark = True
    torch.manual_seed(opt.seed)

    model = cls_model('rsn34', 448, 5, None, None, None)
    criterion = nn.CrossEntropyLoss().cuda()

    train_data_loader = DataLoader(kaggleClsTrain(448, 512, False),
                                   num_workers=2, batch_size=16, shuffle=True, pin_memory=True)
    lr = 0.001
    optimizer = optim.SGD([{'params':model.base.parameters()}, {'params':model.cls.parameters()}],
                          lr=lr)
    model = torch.nn.DataParallel(model).cuda()
    for epoch in range(100):
        cls_train(train_data_loader, model, criterion, optimizer, epoch, False)

if __name__ == '__main__':
    main()



