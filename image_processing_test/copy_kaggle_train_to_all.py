import argparse
import os
from glob import glob
import shutil

parser = argparse.ArgumentParser(description='generate train & validation set')

parser.add_argument('--root', required=True)
parser.add_argument('--src', required=True)
parser.add_argument('--dst', required=True)

args = parser.parse_args()

root = args.root
src = args.src
dst = args.dst

# train_path = os.path.join(root, 'train_images')
# all_path = os.path.join(root, 'all')

src_list = glob(os.path.join(src, '*.png'))

for index in src_list:
    shutil.copy(index, dst)
    print('copy from {0} to {1}'.format(index, dst))