import argparse
import os
from glob import glob
import shutil

parser = argparse.ArgumentParser(description='generate train & validation set')

parser.add_argument('--root', required=True)
parser.add_argument('--testratio', type=float, default=0.1)

args = parser.parse_args()

root = args.root

train_path = os.path.join(root, 'train_images')
all_path = os.path.join(root, 'all')

src_list = glob(os.path.join(train_path, '*.png'))

for index in src_list:
    shutil.copy(index, all_path)
    print('copy from {0} to {1}'.format(index, all_path))