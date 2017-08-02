import json
import torch.utils.data
import numpy as np
from PIL import Image
from utils import TenCrop, HorizontalFlip, Affine, ColorJitter, Lighting, PILColorJitter
import torchvision.transforms as transforms


class kaggleClsTrain(torch.utils.data.Dataset):
	def __init__(self, crop_size, scale_size, baseline):
		super(kaggleClsTrain, self).__init__()
		self.image = ['data/kaggle/train_images/' + line.strip() + '_' + str(scale_size) + '.png' for line in open('data/kaggle/train_images.txt', 'r')]
		self.label = torch.from_numpy(np.array(np.loadtxt('data/kaggle/train_labels.txt'), np.int))
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


class kaggleClsVal(torch.utils.data.Dataset):
	def __init__(self, crop_size, scale_size):
		super(kaggleClsVal, self).__init__()
		self.image = ['data/kaggle/val_images/' + line.strip() + '_' + str(scale_size) + '.png' for line in open('data/kaggle/val_images.txt', 'r')]
		self.label = torch.from_numpy(np.array(np.loadtxt('data/kaggle/val_labels.txt'), np.int))
		with open('data/kaggle/info.json', 'r') as fp:
			info = json.load(fp)
		mean_values = torch.from_numpy(np.array(info['mean'], dtype=np.float32) / 255)
		std_values = torch.from_numpy(np.array(info['std'], dtype=np.float32) / 255)
		self.transform = transforms.Compose([
			transforms.Scale(scale_size),
			transforms.CenterCrop(crop_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean_values, std=std_values),
		])

	def __getitem__(self, index):
		return self.transform(Image.open(self.image[index])), self.label[index]

	def __len__(self):
		if len(self.image) != len(self.label):
			raise Exception("The number of images and labels should be the same.")
		return len(self.label)


class kaggleClsValTenCrop(torch.utils.data.Dataset):
	def __init__(self, crop_idx, crop_size, scale_size):
		super(kaggleClsValTenCrop, self).__init__()
		self.image = ['data/kaggle/val_images/' + line.strip() + '_' + str(scale_size) + '.png' for line in open('data/kaggle/val_images.txt', 'r')]
		with open('data/kaggle/info.json', 'r') as fp:
			info = json.load(fp)
		mean_values = torch.from_numpy(np.array(info['mean'], dtype=np.float32) / 255)
		std_values = torch.from_numpy(np.array(info['std'], dtype=np.float32) / 255)
		self.transform = transforms.Compose([
			TenCrop(crop_idx // 2, crop_size, scale_size),
			HorizontalFlip(crop_idx % 2),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean_values, std=std_values),
		])

	def __getitem__(self, index):
		return self.transform(Image.open(self.image[index]))

	def __len__(self):
		return len(self.image)


class kaggleClsTrain1(torch.utils.data.Dataset):
	def __init__(self, crop_size, scale_size, baseline):
		super(kaggleClsTrain1, self).__init__()
		self.image = ['data/kaggle1/train_images/train/' + line.strip() + '_' + str(scale_size) + '.png' for line in open('data/kaggle1/train_images/train/train_images.txt', 'r')]
		self.label = torch.from_numpy(np.array(np.loadtxt('data/kaggle1/train_images/train/train_labels.txt'), np.int))
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


class kaggleClsVal1(torch.utils.data.Dataset):
	def __init__(self, crop_size, scale_size):
		super(kaggleClsVal1, self).__init__()
		self.image = ['data/kaggle1/train_images/val/' + line.strip() + '_' + str(scale_size) + '.png' for line in open('data/kaggle1/train_images/val/val_images.txt', 'r')]
		self.label = torch.from_numpy(np.array(np.loadtxt('data/kaggle1/train_images/val/val_labels.txt'), np.int))
		with open('data/kaggle/info.json', 'r') as fp:
			info = json.load(fp)
		mean_values = torch.from_numpy(np.array(info['mean'], dtype=np.float32) / 255)
		std_values = torch.from_numpy(np.array(info['std'], dtype=np.float32) / 255)
		self.transform = transforms.Compose([
			transforms.Scale(scale_size),
			transforms.CenterCrop(crop_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean_values, std=std_values),
		])

	def __getitem__(self, index):
		return self.transform(Image.open(self.image[index])), self.label[index]

	def __len__(self):
		if len(self.image) != len(self.label):
			raise Exception("The number of images and labels should be the same.")
		return len(self.label)


class kaggleClsTrain_ZZ(torch.utils.data.Dataset):
	def __init__(self, crop_size, scale_size, baseline):
		super(kaggleClsTrain_ZZ, self).__init__()
		self.image = ['data/zhizhen/train/' + line.strip() + '_' + str(scale_size) + '.png' for line in open('data/zhizhen/train/train_images.txt', 'r')]
		self.label = torch.from_numpy(np.array(np.loadtxt('data/zhizhen/train/train_labels.txt'), np.int))
		with open('data/zhizhen/info.json', 'r') as fp:
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

class kaggleClsVal_ZZ(torch.utils.data.Dataset):
	def __init__(self, crop_size, scale_size):
		super(kaggleClsVal_ZZ, self).__init__()
		self.image = ['data/zhizhen/val/' + line.strip() + '_' + str(scale_size) + '.png' for line in open('data/zhizhen/val/val_images.txt', 'r')]
		self.label = torch.from_numpy(np.array(np.loadtxt('data/zhizhen/val/val_labels.txt'), np.int))
		with open('data/zhizhen/info.json', 'r') as fp:
			info = json.load(fp)
		mean_values = torch.from_numpy(np.array(info['mean'], dtype=np.float32) / 255)
		std_values = torch.from_numpy(np.array(info['std'], dtype=np.float32) / 255)
		self.transform = transforms.Compose([
			transforms.Scale(scale_size),
			transforms.CenterCrop(crop_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean_values, std=std_values),
		])

	def __getitem__(self, index):
		return self.transform(Image.open(self.image[index])), self.label[index]

	def __len__(self):
		if len(self.image) != len(self.label):
			raise Exception("The number of images and labels should be the same.")
		return len(self.label)


class kaggleClsTest_ZZ(torch.utils.data.Dataset):
	def __init__(self, crop_size, scale_size):
		super(kaggleClsTest_ZZ, self).__init__()
		self.image = ['data/zhizhen/test/' + line.strip() + '_' + str(scale_size) + '.png' for line in open('data/zhizhen/test/test_images.txt', 'r')]
		self.label = torch.from_numpy(np.array(np.loadtxt('data/zhizhen/test/test_labels.txt'), np.int))
		with open('data/zhizhen/info.json', 'r') as fp:
			info = json.load(fp)
		mean_values = torch.from_numpy(np.array(info['mean'], dtype=np.float32) / 255)
		std_values = torch.from_numpy(np.array(info['std'], dtype=np.float32) / 255)
		self.transform = transforms.Compose([
			transforms.Scale(scale_size),
			transforms.CenterCrop(crop_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean_values, std=std_values),
		])

	def __getitem__(self, index):
		return self.transform(Image.open(self.image[index])), self.label[index]

	def __len__(self):
		if len(self.image) != len(self.label):
			raise Exception("The number of images and labels should be the same.")
		return len(self.label)