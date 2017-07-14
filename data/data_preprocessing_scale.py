from PIL import Image
import os

scale_sizes = [1024, 512, 256, 128]

def scale_image(root, imglist, threadid):
    print('===>begin: ', str(threadid))
    print(imglist)
    for index in imglist:
        image = Image.open(os.path.join(root, '{}.jpg'.format(index)))
        w, h = image.size
        tw, th = (min(w, h), min(w, h))
        image = image.crop((w // 2 - tw // 2, h // 2 - th // 2, w // 2 + tw // 2, h // 2 + th // 2))
        w, h = image.size
        for scale_size in scale_sizes:
            tw, th = (scale_size, scale_size)
            ratio = tw / w
            assert ratio == th / h
            if ratio < 1:
                image = image.resize((tw, th), Image.ANTIALIAS)
            elif ratio > 1:
                image = image.resize((tw, th), Image.CUBIC)
            image.save(os.path.join(root, '{0}/{1}_{2}.png'.format(scale_size, index, scale_size)))
            print('save: ', os.path.join(root, '{0}/{1}_{2}.png'.format(scale_size, index, scale_size)))
    print('===>end: ', str(threadid))


import threading

import argparse

parser = argparse.ArgumentParser(description='Diabetic Retinopathy classification preprocessed')

parser.add_argument('--root', required=True)
parser.add_argument('--workers', type=int, default=1)

args = parser.parse_args()

assert os.path.isdir(args.root) == True

root = args.root

for scale_size in scale_sizes:
    if not os.path.exists(os.path.join(root, str(scale_size))):
        os.mkdir(os.path.join(root, str(scale_size)))

from glob import glob

imagelist = [os.path.basename(f).split('.')[0] for f in glob(os.path.join(root, '*.jpg'))]

import math

num = math.ceil(len(imagelist) / args.workers)
thread_num = args.workers
threads = []
for i in range(thread_num):
    thread_imagelist = imagelist[i*num:min((i+1)*num, len(imagelist))]
    t = threading.Thread(target=scale_image, args=(root, thread_imagelist, i))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

