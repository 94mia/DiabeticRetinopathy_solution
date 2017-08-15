import os
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='generate dr data flag files')
parser.add_argument('--root', required=True)
args = parser.parse_args()

# root = '/Users/zhangweidong03/data/zhizhen_cls'

root = args.root

images_list = glob(os.path.join(root, '*.jpg'))

names_list = []
flags_list = []

for index in images_list:
    name = os.path.basename(index.split('.')[0])
    print(name)
    flag = name.split('_')[4]
    names_list.append(name)
    flags_list.append(flag)

print(flags_list)
print(names_list)

assert len(names_list) == len(flags_list)

images_file_path = os.path.join(root, 'train_images.txt')
labels_file_path = os.path.join(root, 'train_labels.txt')

with open(images_file_path, 'w') as f:
    for index in names_list:
        f.write(index+'\r')

with open(labels_file_path, 'w') as f:
    for index in flags_list:
        f.write(str(index)+'\r')