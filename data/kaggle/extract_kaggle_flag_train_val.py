'''
This module is used to extract kaggle level flags to two catagories: dr level, referable level
'''

import pandas as pd
import numpy as np
import argparse
import os

from sklearn.cross_validation import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description='extract kaggle flags')
    parser.add_argument('--root', required=True)
    parser.add_argument('--imagestxt', default='train_images.txt')
    parser.add_argument('--labelstxt', default='train_labels.txt')
    parser.add_argument('--csv', default='data_multi.csv')

    parser.add_argument('--csvtrain', default='train_multi.csv')
    parser.add_argument('--csvval', default='val_multi.csv')
    parser.add_argument('--csvtest', default='test_multi.csv')
    parser.add_argument('--valratio', type=float, default=0.1)
    parser.add_argument('--testratio', type=float, default=0.2)

    return parser.parse_args()

args = parse_args()

print(args)

root = args.root
imagestxt = args.imagestxt
labelstxt = args.labelstxt

imagestxt = os.path.join(root, imagestxt)
labelstxt = os.path.join(root, labelstxt)

images_list = [line.strip() for line in open(imagestxt)]
labels_list = [int(line.strip()) for line in open(labelstxt)]
referable_list = [1 if x>1 else 0 for x in labels_list]

Auxiliary_list = [0]*len(images_list)

data = np.column_stack((images_list, labels_list, Auxiliary_list, referable_list))

df = pd.DataFrame(data, columns=['images', 'dr_level', 'dme_level','referable'])

kaggle_flags_folder = os.path.join(root, 'flags')

if not os.path.isdir(kaggle_flags_folder):
    os.mkdir(kaggle_flags_folder)

df.to_csv(os.path.join(args.root, args.csv))


# try to read

dfs = pd.read_csv(os.path.join(args.root, args.csv))

for index, row in df.iterrows():
    print(row)

print(dfs)

csv_train = os.path.join(args.root, args.csvtrain)
csv_val = os.path.join(args.root, args.csvval)
# csv_test = os.path.join(args.root, args.csvtest)
#
# train_data, test_data = train_test_split(data, test_size=args.testratio)
train_data, val_data = train_test_split(data, test_size=args.valratio)
#
train_df = pd.DataFrame(train_data, columns=['images', 'dr_level', 'dme_level','referable'])
val_df = pd.DataFrame(val_data, columns=['images', 'dr_level', 'dme_level','referable'])
# test_df = pd.DataFrame(test_data, columns=['image', 'dr_level', 'dme_level'])
#
train_df.to_csv(csv_train)
val_df.to_csv(csv_val)
# test_df.to_csv(csv_test)


