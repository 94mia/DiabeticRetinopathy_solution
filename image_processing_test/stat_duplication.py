import argparse
import os
from glob import glob

parser = argparse.ArgumentParser(description='generate train & validation set')

parser.add_argument('--root', required=True)
parser.add_argument('--testratio', type=float, default=0.1)

args = parser.parse_args()

root = args.root

folder0 = os.path.join(root, '0')
folder1 = os.path.join(root, '1')
folder2 = os.path.join(root, '2')
folder3 = os.path.join(root, '3')
folder4 = os.path.join(root, '4')

images0 = glob(os.path.join(folder0,'*.jpg'))
images1 = glob(os.path.join(folder1,'*.jpg'))
images2 = glob(os.path.join(folder2,'*.jpg'))
images3 = glob(os.path.join(folder3,'*.jpg'))
images4 = glob(os.path.join(folder4,'*.jpg'))

images0 = [os.path.basename(f) for f in images0]
images1 = [os.path.basename(f) for f in images1]
images2 = [os.path.basename(f) for f in images2]
images3 = [os.path.basename(f) for f in images3]
images4 = [os.path.basename(f) for f in images4]

cnt = 0

for index in images1:
    if index in images0:
        cnt += 1

print('duplicate in 1 count: {}'.format(cnt))

cnt = 0

for index in images2:
    if index in images0:
        cnt += 1

print('duplicate in 2 count: {}'.format(cnt))

cnt = 0

for index in images3:
    if index in images0:
        cnt += 1

print('duplicate in 3 count: {}'.format(cnt))

cnt = 0

for index in images4:
    if index in images0:
        cnt += 1

print('duplicate in 4 count: {}'.format(cnt))