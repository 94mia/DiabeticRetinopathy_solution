from glob import glob
import os
import argparse

parser = argparse.ArgumentParser(description='rename dr-0 level')

parser.add_argument('--root', required=True)

args = parser.parse_args()

root = args.root

images_list = glob(os.path.join(root, '*.jpg'))

for index in images_list:
    #os.rename(index, os.path.join(root, 'dr0_' + os.path.basename(index).split('(')[1].split(')')[0]+'.jpg'))
    os.rename(index, os.path.join(root, 'dr0_' + os.path.basename(index)))

