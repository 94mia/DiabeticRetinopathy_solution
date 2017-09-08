# import torch
#
# from pycocotools.coco import COCO
#
# import os
#
# from PIL import Image
#
# from torch.utils.data import DataLoader
#
# from torchvision.transforms import ToTensor, Compose
#
# import torchvision.transforms as transforms
#
# class CocoDetection(torch.utils.data.Dataset):
#     def __init__(self, root, annfile, scale_size=None, transform=None):
#         self.root = root
#         self.annfile = annfile
#         self.coco = COCO(os.path.join(root, annfile))
#         self.ids = list(self.coco.imgs.keys())
#         self.transform = transform
#         self.scale_size = scale_size
#         self.ord2cid = sorted(self.coco.cats.keys())
#         self.cid2ord = {i:o for o,i in enumerate(self.ord2cid)}
#
#     def __getitem__(self, item):
#         img_id = self.ids[item]
#         ann_ids = self.coco.getAnnIds(imgIds=img_id)
#         anns = self.coco.loadAnns(ann_ids)
#
#         filename = self.coco.loadImgs(img_id)[0]['file_name']
#         filename = os.path.join(self.root, 'train2014/'+filename)
#         assert os.path.isfile(filename)
#         img = Image.open(filename)
#
#         for ann in anns:
#             ann['bbox'][2] += ann['bbox'][0]
#             ann['bbox'][3] += ann['bbox'][1]
#             ann['ordered_id'] = self.cid2ord[ann['category_id']]
#             ann['scale_ratio'] = 1
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, anns
#
#     def __len__(self):
#         return len(self.ids)
#
#
# # root = '/home/weidong/code/github/mask_rcnn_pytorch/data/COCO'
# # annfile = 'annotations/instances_train2014.json'
#
# def dataset_test():
#     root = '/home/weidong/code/github/mask_rcnn_pytorch/data/COCO'
#     annfile = 'annotations/instances_train2014.json'
#
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#
#     trans = transforms.Compose([transforms.ToTensor(), normalize])
#
#     dataloader = DataLoader(CocoDetection(root, annfile, transform=trans))
#
#     for i,(img,anns) in enumerate(dataloader):
#         print(anns)
#
# if __name__ == '__main__':
#     dataset_test()


x = 0.12341234123

info = 'sen: {0:.4f}'.format(x)

print(info)
