import argparse

parser = argparse.ArgumentParser(description='ex segmentation with k-means')
parser.add_argument('--root', required=True)

args = parser.parse_args()

root = args.root

import os

if not os.path.exists(root):
    print('{} is not exist!'.format(root))

images_path = os.path.join(root, 'images')

if not os.path.exists(images_path):
    print('{} is not exist!'.format(images_path))

labels_kmeans_path = os.path.join(root, 'labels_kmeans')

if not os.path.exists(labels_kmeans_path):
    print('{0} not exists! create!'.format(labels_kmeans_path))
    os.mkdir(labels_kmeans_path)

from glob import glob

images_list = glob(os.path.join(images_path, '*.png'))


import numpy as np
import cv2
from PIL import Image

# for index in images_list:
#     img = cv2.imread(index)
for index in range(1):
    img = cv2.imread('/Users/zhangweidong03/data/ex/ex_patches/images/C0010492_EX_1_5.png')
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    im = gray_image = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    thresh, im = cv2.threshold(im, 65, 255, cv2.THRESH_BINARY)
    pil_img = Image.fromarray(im)
    pil_img.show()

    pil_img_bgr = Image.fromarray(res2)
    pil_img_bgr.show()
    pil_img_raw = Image.open('/Users/zhangweidong03/data/ex/ex_patches/images/C0010492_EX_1_5.png')
    pil_img_raw.show()