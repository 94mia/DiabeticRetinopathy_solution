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


def parse_args():
	parser = argparse.ArgumentParser(description='Semantic Segmentation by Dilated ConvNet')
	parser.add_argument('--exp', required=True, help='The name of experiment')
	parser.add_argument('--dataset', default='kaggle', choices=['kaggle'], help='The dataset to use')
	parser.add_argument('--phase', default='train', help='The phase to run')
	parser.add_argument('--output', default='output', help='The output dir')
	parser.add_argument('--model', default='drn42', choices=['drn26', 'drn42', 'drn58', 'rsn50', 'rsn34'])
	parser.add_argument('--batch', default=8, type=int, help='The batch size of training')
	parser.add_argument('--crop', default=224, type=int, help='The crop size of input')
	parser.add_argument('--size', default=256, type=int, choices=[128, 256, 512, 1024], help='The rescaled size of input')
	parser.add_argument('--weight', default=None, help='The path of pretrained model')
	parser.add_argument('--lr', default=0.01, type=float, help='The initial learning rate')
	parser.add_argument('--fix', default=200, type=float, help='Fixing learning rate during training')
	parser.add_argument('--mom', default=0.9, type=float, help='The momentum of SGD')
	parser.add_argument('--wd', default=1e-4, type=float, help='The weight decay of SGD')
	parser.add_argument('--step', default=129, type=int, help='The step size of training')
	parser.add_argument('--epoch', default=233, type=int, help='The number of training epoch')
	parser.add_argument('--display', default=10, type=int, help='The frequency of printing log')
	parser.add_argument('--seed', default=111, type=int, help='Random seed to use')
	parser.add_argument('--threads', default=4, type=int, help='The number of threads for data loader to use')
	parser.add_argument('--baseline', action='store_true', help='Enable baseline augmentation')
	parser.add_argument('--balance', action='store_true', help='Enable weight-balanced loss')
	parser.add_argument('--tencrop', action='store_true', help='Enable ten-crop test')
	parser.add_argument('--cls2reg', action='store_true', help='Use regression instead of classification')
	parser.add_argument('--scratch', action='store_true', help='Enable from-the-scatch training')
	return parser.parse_args()


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


def cls_train(train_data_loader, model, criterion, optimizer, epoch, display):
	model.train()
	tot_pred = np.array([], dtype=int)
	tot_label = np.array([], dtype=int)
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	end = time.time()
	logger = []
	for num_iter, (image, label) in enumerate(train_data_loader):
		data_time.update(time.time() - end)
		final = model(Variable(image))
		loss = criterion(final, Variable(label.cuda()))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		batch_time.update(time.time() - end)
		_, pred = torch.max(final, 1)
		pred = pred.cpu().data.numpy().squeeze()
		label = label.cpu().numpy().squeeze()
		tot_pred = np.append(tot_pred, pred)
		tot_label = np.append(tot_label, label)
		kappa = quadratic_weighted_kappa(tot_label, tot_pred)
		losses.update(loss.data[0], image.size(0))
		end = time.time()
		if num_iter % display == 0:
			print('Epoch: [{0}][{1}/{2}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
			      'Data {data_time.avg:.3f}\t' 'Loss {loss.avg:.4f}\t'  'Kappa {kappa:.4f}\t'
			      .format(epoch, num_iter, len(train_data_loader), batch_time=batch_time, data_time=data_time, loss=losses, kappa=kappa))
			# 'Accuracy {accuracy:.2f}\t' accuracy=100 * (tot_pred == tot_label).sum() / len(tot_label) Not good for unbalanced classes
			logger.append('Epoch: [{0}][{1}/{2}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
			              'Data {data_time.avg:.3f}\t' 'Loss {loss.avg:.4f}\t'  'Kappa {kappa:.4f}\t'
			              .format(epoch, num_iter, len(train_data_loader), batch_time=batch_time, data_time=data_time, loss=losses, kappa=kappa))
	return logger


def cls_val(eval_data_loader, model, criterion, ten_crop_data_loader):
	model.eval()
	tot_pred = np.array([], dtype=int)
	tot_label = np.array([], dtype=int)
	losses = AverageMeter()
	batch_time = AverageMeter()
	data_time = AverageMeter()
	end = time.time()
	logger = []
	for num_iter, (image, label) in enumerate(eval_data_loader):
		data_time.update(time.time() - end)
		final = model(Variable(image, requires_grad=False, volatile=True))
		if ten_crop_data_loader:
			for cropped_data_loader in ten_crop_data_loader:
				cropped_image = next(cropped_data_loader)
				final += model(Variable(cropped_image, requires_grad=False, volatile=True))
			final /= 11
		loss = criterion(final, Variable(label.cuda()))
		_, pred = torch.max(final, 1)
		pred = pred.cpu().data.numpy().squeeze()
		label = label.cpu().numpy().squeeze()
		tot_pred = np.append(tot_pred, pred)
		tot_label = np.append(tot_label, label)
		losses.update(loss.data[0], image.size(0))
		kappa = quadratic_weighted_kappa(tot_label, tot_pred)
		batch_time.update(time.time() - end)
		end = time.time()
		print('Eval: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
		      'Data {data_time.avg:.3f}\t' 'Loss {loss.avg:.4f}\t'  'Kappa {kappa:.4f}\t'
		      .format(num_iter, len(eval_data_loader), batch_time=batch_time, data_time=data_time, loss=losses, kappa=kappa))
		logger.append('Eval: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
			  'Data {data_time.avg:.3f}\t' 'Loss {loss.avg:.4f}\t'  'Kappa {kappa:.4f}\t'
			  .format(num_iter, len(eval_data_loader), batch_time=batch_time, data_time=data_time, loss=losses,
					  kappa=kappa))

	return kappa, tot_pred, tot_label, logger


class Cls2Reg(nn.Module):
	def __init__(self, size_average=False):
		super(Cls2Reg, self).__init__()
		self.loss = nn.SmoothL1Loss(size_average=size_average).cuda()
		#self.loss = nn.MSELoss(size_average=size_average).cuda()

	def forward(self, final, label):
		return self.loss(final, label.float())


def main():
	print('===> Parsing options')
	opt = parse_args()
	print(opt)
	cudnn.benchmark = True
	torch.manual_seed(opt.seed)
	if not torch.cuda.is_available():
		raise Exception("No GPU found")
	if not os.path.exists(opt.output):
		os.makedirs(opt.output)
	time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
	output_dir = os.path.join(opt.output, opt.dataset + '_cls_' + opt.phase + '_' + time_stamp + '_' + opt.model + '_' + opt.exp)
	if not os.path.exists(output_dir):
		print('===> Creating ', output_dir)
		os.makedirs(output_dir)

	print('===> Building model')
	with open('data/' + opt.dataset + '/info.json', 'r') as fp:
		info = json.load(fp)
	num_classes = np.int(info['classes'])
	name_classes = np.array(info['label'], dtype=np.str)
	w_i = np.array(info['weights_initial'], dtype=np.float32)
	w_f = np.array(info['weights_final'], dtype=np.float32)
	w_r = np.float32(info['weights_ratio'])
	model = cls_model(opt.model, opt.crop, num_classes, opt.weight, opt.scratch, opt.cls2reg)
	if opt.cls2reg:
		criterion = Cls2Reg(size_average=False)
	else:
		criterion = nn.CrossEntropyLoss().cuda()

	if opt.phase == 'train':
		print('===> Training model')
		train_data_loader = DataLoader(dataset=globals()[opt.dataset + 'ClsTrain_ZZ'](crop_size=opt.crop, scale_size=opt.size, baseline=opt.baseline),
		                               num_workers=opt.threads, batch_size=opt.batch, shuffle=True, pin_memory=True)
		val_data_loader1 = DataLoader(dataset=globals()[opt.dataset + 'ClsVal_ZZ'](crop_size=opt.crop, scale_size=opt.size),
									 num_workers=opt.threads, batch_size=opt.batch, shuffle=False, pin_memory=False)
		kappa_best = 0
		for epoch in range(opt.epoch):
			if opt.balance:
				w_epoch = torch.from_numpy(w_i*w_r**epoch + w_f*(1-w_r**epoch))
				criterion = nn.CrossEntropyLoss(weight=w_epoch).cuda()
			if epoch < opt.fix:
				lr = opt.lr
			else:
				if opt.baseline:
					lr = opt.lr / 10
				else:
					lr = opt.lr * (0.1 ** (epoch//opt.step))
					#lr = max((1 - float(epoch - opt.fix) / (opt.epoch - opt.fix)) ** 0.9 * opt.lr, 1e-6)
			optimizer = optim.SGD([{'params': model.base.parameters()}, {'params': model.cls.parameters()}], lr=lr, momentum=opt.mom, weight_decay=opt.wd, nesterov=True)
			logger = cls_train(train_data_loader, torch.nn.DataParallel(model).cuda(), criterion, optimizer, epoch, opt.display)
			ten_crop_data_loader = []
			kappa, pred, label,  logger_val= cls_val(val_data_loader1, torch.nn.DataParallel(model).cuda(), criterion,
										 ten_crop_data_loader)
			if kappa > kappa_best:
				kappa_best = kappa
				torch.save(model.cpu().state_dict(), os.path.join(output_dir, opt.dataset + '_cls_' + opt.model + '_%03d' % epoch + '_best.pth'))
			print('current best kappa: '.format(kappa_best))
			# torch.save(model.cpu().state_dict(), os.path.join(output_dir, opt.dataset + '_cls_' + opt.model + '_%03d' % epoch + '.pth'))
			print('===> ' + output_dir + '/' + opt.dataset + '_cls_' + opt.model + '_%03d' % epoch + '.pth')
			if not os.path.isfile(os.path.join(output_dir, 'train.log')):
				with open(os.path.join(output_dir, 'train.log'), 'w') as fp:
					fp.write(str(opt) + '\n\n')
			with open(os.path.join(output_dir, 'train.log'), 'a') as fp:
				fp.write('\n' + '\n'.join(logger))
				fp.write('\n' + '\n'.join(logger_val))

	elif opt.phase == 'val':
		if opt.weight:
			print('===> Evaluating model')
			val_data_loader = DataLoader(dataset=globals()[opt.dataset + 'ClsTest_ZZ'](crop_size=opt.crop, scale_size=opt.size),
			                             num_workers=opt.threads, batch_size=opt.batch, shuffle=False, pin_memory=False)
			ten_crop_data_loader = []
			if opt.tencrop:
				for crop_idx in range(10):
					ten_crop_data_loader.append(iter(DataLoader(dataset=globals()[opt.dataset + 'ClsValTenCrop'](crop_idx, opt.crop, opt.size),
					                                            num_workers=opt.threads, batch_size=opt.batch, shuffle=False, pin_memory=False)))
			kappa, pred, label, _ = cls_val(val_data_loader, torch.nn.DataParallel(model).cuda(), criterion, ten_crop_data_loader)
			print('===> Kappa: %.4f' % kappa)
			print('===> Confusion Matrix:')
			print(confusion_matrix(label, pred))
			with open(os.path.join(output_dir, 'val.log'), 'w') as fp:
				fp.write(str(opt)+'\n')
				fp.write('\n===> Kappa: %.4f' % kappa)
			np.save(os.path.join(output_dir, 'results.npy'), pred, label, confusion_matrix(label, pred))

		else:
			raise Exception('No weights found')

	else:
		raise Exception('No phase found')


if __name__ == '__main__':
	main()
