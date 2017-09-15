import os
import argparse

import shutil

import pandas as pd

from glob import glob

from sklearn.cross_validation import train_test_split

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='zhizhen data processing')
    parser.add_argument('--root', default='./LabelImages')

    parser.add_argument('--csvdata', default='data_multi.csv')
    parser.add_argument('--csvtrain', default='train_multi.csv')
    parser.add_argument('--csvval', default='val_multi.csv')
    parser.add_argument('--csvtest', default='test_multi.csv')
    parser.add_argument('--csvtrain_bin', default='train_bin.csv')
    parser.add_argument('--csvval_bin', default='val_bin.csv')
    parser.add_argument('--csvtest_bin', default='test_bin.csv')
    parser.add_argument('--valratio', type=float, default=0.1)
    parser.add_argument('--testratio', type=float, default=0.2)

    return parser.parse_args()

args = parse_args()

assert os.path.isdir(args.root)

new_folder = os.path.join(args.root, 'abnormal')

os.makedirs(new_folder, exist_ok=True)

assert os.path.isdir(new_folder)

images_list = []
dr_list = []
dme_list = []

with open('/home/weidong/code/github/DiabeticRetinopathy_solution/data/zhizhen_new/LabelImages/label.txt', 'r') as f:
    line = f.readline()
    lines = f.readlines()
    for line in lines:
        vec = line.strip().split(' ')
        dr_level = int(vec[1])
        dme_level = int(vec[2])
        if (dr_level <= 1 and dme_level > 0) or (dr_level == 0):
            continue
        images_list.append(vec[0])
        dr_list.append(dr_level)
        dme_list.append(dme_level)
        # shutil.copy(os.path.join(args.root, '{0}/{1}'.format(dr_level, vec[0])), new_folder)
        print('copy from {0} to {1}'.format(os.path.join(args.root, '{0}/{1}'.format(dr_level, vec[0])),new_folder))

normal_list = glob('./NormalData/*.jpg')
for index in normal_list:
    images_list.append(os.path.basename(index))
    dr_list.append(0)
    dme_list.append(0)

    # shutil.copy(index, new_folder)
    print('copy from {0} to {1}'.format(index, new_folder))

images_list = [index.split('.')[0] for index in images_list]

randindex = np.random.permutation(len(images_list))

images_list = [images_list[i] for i in randindex]
dr_list = [dr_list[i] for i in randindex]
dme_list = [dme_list[i] for i in randindex]

assert len(images_list) == len(dr_list) == len(dme_list)

print('before: {}'.format(len(images_list)))

all_list = glob(os.path.join(args.root, '512/*.png'))
all_list = [os.path.basename(index).replace('_512.png', '') for index in all_list]

err_list = []

for i, index in enumerate(images_list):
    if index in all_list:
        continue
    err_list.append(i)

print('err list len: {}'.format(len(err_list)))

for index in err_list:
    images_list.remove(images_list[index])
    dr_list.remove(dr_list[index])
    dme_list.remove(dme_list[index])

assert len(images_list) == len(dr_list) == len(dme_list)

print('after: {}'.format(len(images_list)))

bin_list = []
for i in range(len(dr_list)):
    # if (dr_list[i] > 2) or (dme_list[i] > 0 and dr_list[i] == 2):
    if (dr_list[i] > 1) or (dme_list[i] > 0):
        bin_list.append(1)
    else:
        bin_list.append(0)

data = np.column_stack((images_list, dr_list, dme_list, bin_list))

train_data, test_data = train_test_split(data, test_size=args.testratio)
train_data, val_data = train_test_split(train_data, test_size=args.valratio)


data_df = pd.DataFrame(data, columns=['image', 'dr_level', 'dme_level', 'totreat'])
train_df = pd.DataFrame(train_data, columns=['image', 'dr_level', 'dme_level', 'totreat'])
val_df = pd.DataFrame(val_data, columns=['image', 'dr_level', 'dme_level', 'totreat'])
test_df = pd.DataFrame(test_data, columns=['image', 'dr_level', 'dme_level', 'totreat'])

csv_train = os.path.join(args.root, args.csvtrain)
csv_val = os.path.join(args.root, args.csvval)
csv_test = os.path.join(args.root, args.csvtest)

csv_data = os.path.join(args.root, args.csvdata)

train_df.to_csv(csv_train)
val_df.to_csv(csv_val)
test_df.to_csv(csv_test)
data_df.to_csv(csv_data)