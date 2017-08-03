import argparse
import os
from glob import glob
import shutil

parser = argparse.ArgumentParser(description='merge kaggle train and test set txt files')

parser.add_argument('--root', required=True)
parser.add_argument('--rootk', required=True)
parser.add_argument('--rootz', required=True)
parser.add_argument('--rootm', required=True)

args = parser.parse_args()

root = args.root
rootk = args.rootk
rootz = args.rootz
rootm = args.rootm

k_images_txt = os.path.join(rootk, 'val_images.txt')
k_labels_txt = os.path.join(rootk, 'val_labels.txt')

z_images_txt = os.path.join(rootz, 'val_images.txt')
z_labels_txt = os.path.join(rootz, 'val_labels.txt')

m_images_txt = os.path.join(rootm, 'val_images.txt')
m_labels_txt = os.path.join(rootm, 'val_labels.txt')

assert os.path.isfile(k_images_txt)
assert os.path.isfile(k_labels_txt)
assert os.path.isfile(z_images_txt)
assert os.path.isfile(z_labels_txt)

k_images_list = [l.strip() for l in open(k_images_txt, 'r')]
k_labels_list = [l.strip() for l in open(k_labels_txt, 'r')]

assert len(k_images_list) == len(k_labels_list)

z_images_list = [l.strip() for l in open(z_images_txt, 'r')]
z_labels_list = [l.strip() for l in open(z_labels_txt, 'r')]

assert len(z_images_list) == len(z_labels_list)

print('kaggle list count is: {}'.format(len(k_images_list)))
print('zz list count is: {}'.format(len(z_images_list)))

a_images_list = k_images_list
a_images_list[len(k_images_list):len(k_images_list)] = z_images_list

a_labels_list = k_labels_list
a_labels_list[len(k_labels_list):len(k_labels_list)] = z_labels_list

print('merge list count is: {}'.format(len(a_images_list)))


assert len(a_images_list) == len(k_images_list) + len(z_images_list)
assert len(a_labels_list) == len(k_labels_list) + len(z_labels_list)
assert len(a_images_list) == len(a_labels_list)



with open(m_images_txt, 'w') as f:
    for index in a_images_list:
        f.write(index+'\r')

with open(m_labels_txt, 'w') as f:
    for index in a_labels_list:
        f.write(index+'\r')

print('finish')
