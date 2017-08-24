'''
This module is used to extract kaggle level flags to two catagories: dr level, referable level
'''

import pandas as pd
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='extract kaggle flags')
    parser.add_argument('--root', required=True)
    parser.add_argument('--imagestxt', required=True)
    parser.add_argument('--labelstxt', required=True)
    parser.add_argument('--csvfile', required=True)

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

data = np.column_stack((images_list, labels_list, referable_list))

df = pd.DataFrame(data, columns=['images', 'level', 'totreat'])

kaggle_flags_folder = os.path.join(root, 'flags')

if not os.path.isdir(kaggle_flags_folder):
    os.mkdir(kaggle_flags_folder)

df.to_csv(os.path.join(kaggle_flags_folder, args.csvfile))


# try to read

dfs = pd.read_csv(os.path.join(kaggle_flags_folder, args.csvfile))

for index, row in df.iterrows():
    print(row)

print(dfs)

