import pandas as pd

import shutil
import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='dr classification val level comparing files extraction')
    parser.add_argument('--root', required=True)
    parser.add_argument('--csvfile', required=True)

    return parser.parse_args()

args = parse_args()

extracted_path = os.path.join(args.root, 'extraction')

if not os.path.isdir(extracted_path):
    os.mkdir(extracted_path)

for i in range(5):
    for j in range(5):
        folder = os.path.join(extracted_path, 'gt_'+str(i)+'_pred_'+str(j))
        os.makedirs(folder, exist_ok=True)

config = args.csvfile

df = pd.DataFrame.from_csv(config)

images_list = []

for index, row in df.iterrows():
    images_list.append(row)

for index in images_list:
    shutil.copy(index[0], './extraction/gt_'+str(index[1])+'_pred_'+str(index[2]))
