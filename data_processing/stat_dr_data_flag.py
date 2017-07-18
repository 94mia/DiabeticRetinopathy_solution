import os
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='generate dr data flag files')
parser.add_argument('--root', required=True)
args = parser.parse_args()

# root = '/Users/zhangweidong03/data/zhizhen_cls'

root = args.root

dict = {}
list = []

for i in range(5):
    sub_root = os.path.join(root, str(i))
    images_list = glob(os.path.join(sub_root, '*.jpg'))
    for index in images_list:
        dict[index] = i
        list.append(index)

print(dict)

list.sort()

print('====================>')

dict_sort = {}

for index in list:
    dict_sort[index] = dict[index]

print(dict_sort)

# with [open(os.path.join(root, 'train_images.txt'), 'w'), open(os.path.join(root, 'train_labels.txt'), 'w')] as [f1, f2]:

f1 = open(os.path.join(root, 'train_images.txt'), 'w')
f2 = open(os.path.join(root, 'train_labels.txt'), 'w')

for index in list:
    f1.write(os.path.basename(index).split('.')[0]+'\r')
    f2.write(str(dict[index])+'\r')

f1.close()
f2.close()
