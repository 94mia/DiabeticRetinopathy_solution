import argparse
import os
from glob import glob
import shutil

parser = argparse.ArgumentParser(description='merge kaggle train and test set txt files')

parser.add_argument('--root', required=True)

args = parser.parse_args()

root = args.root

train_images_txt = os.path.join(root, 'train_images.txt')
train_labels_txt = os.path.join(root, 'train_labels.txt')
val_images_txt = os.path.join(root, 'val_images.txt')
val_labels_txt = os.path.join(root, 'val_labels.txt')

all_images_txt = os.path.join(root, 'all_images.txt')
all_labels_txt = os.path.join(root, 'all_labels.txt')

assert os.path.isfile(train_images_txt)
assert os.path.isfile(train_labels_txt)
assert os.path.isfile(val_images_txt)
assert os.path.isfile(val_labels_txt)

train_images_list = [line.strip() for line in open(train_images_txt, 'r')]
train_labels_list = [line.strip() for line in open(train_labels_txt, 'r')]

assert len(train_images_list) == len(train_labels_list)

val_images_list = [line.strip() for line in open(val_images_txt, 'r')]
val_labels_list = [line.strip() for line in open(val_labels_txt, 'r')]

assert len(val_images_list) == len(val_labels_list)

all_images_list = train_images_list
all_images_list[len(train_images_list):len(train_images_list)] = val_images_list

all_labels_list = train_labels_list
all_labels_list[len(train_labels_list):len(train_labels_list)] = val_labels_list

assert len(all_images_list) == len(all_labels_list)

print('train list count: {}'.format(len(train_images_list)))
print('val list count: {}'.format(len(val_images_list)))
print('all list count: {}'.format(len(all_images_list)))

with open(all_images_txt, 'w') as f:
    for index in all_images_list:
        f.write(all_images_list[index])

with open(all_labels_txt, 'w') as f:
    for index in all_labels_list:
        f.write((all_labels_list[index]))

