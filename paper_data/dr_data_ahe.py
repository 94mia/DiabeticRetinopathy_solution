'''
note:
1. the root should contained images, labels
2. the output ahe images will also place in root path, like this:
root->
    images
    labels
    ahe_images
'''


import numpy as np

from skimage.filters import threshold_otsu
from skimage import measure, exposure

import skimage

import scipy.misc
from PIL import Image

__all__ = [
    'tight_crop',
    'channelwise_ahe',
]

scales=[512]

def tight_crop(img, size=None):
    img_gray = np.mean(img, 2)
    img_bw = img_gray > threshold_otsu(img_gray)
    img_label = measure.label(img_bw, background=0)
    largest_label = np.argmax(np.bincount(img_label.flatten())[1:])+1

    img_circ = (img_label == largest_label)
    img_xs = np.sum(img_circ, 0)
    img_ys = np.sum(img_circ, 1)
    xs = np.where(img_xs>0)
    ys = np.where(img_ys>0)
    x_lo = np.min(xs)
    x_hi = np.max(xs)
    y_lo = np.min(ys)
    y_hi = np.max(ys)
    img_crop = img[y_lo:y_hi, x_lo:x_hi, :]

    return img_crop


# adaptive historgram equlization
def channelwise_ahe(img):
    img_ahe = img.copy()
    for i in range(img.shape[2]):
        img_ahe[:,:,i] = exposure.equalize_adapthist(img[:,:,i], clip_limit=0.03)
    return img_ahe



import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Microaneurysm data processing')
    parser.add_argument('--root', required=True, help='the root path include following path: images, labels, ...')

    return parser.parse_args()

opt = arg_parse()
print(opt)

import os

if not os.path.isdir(opt.root):
    print('directory error! the path {} is not exist!'.format(opt.root))

images_path = os.path.join(opt.root, str(scales[0]))
ahe_images_path = os.path.join(opt.root, str(scales[0])+'_ahe')

if not os.path.isdir(images_path):
    print('there are no images to be preprocessed!')

if not os.path.isdir(ahe_images_path):
    print('create the output ahe images directory: {}'.format(ahe_images_path))
    os.mkdir(ahe_images_path)

from glob import glob

images_list = glob(os.path.join(images_path, '*.png'))
# images_list = glob(os.path.join(images_path, '*.jpg'))

for image in images_list:
    base_str = os.path.basename(image).split('.')[0]
    output_file = os.path.join(ahe_images_path, base_str+'_ahe.png')
    img = scipy.misc.imread(image)
    img = img.astype(np.float32)
    img /= 255
    img_ahe = channelwise_ahe(img)
    pilImage = Image.fromarray(skimage.util.img_as_ubyte(img_ahe))
    pilImage.save(output_file)
    print('{0} is preprocessed and saved to {1}'.format(image, output_file))