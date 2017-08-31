import pandas as pd

import shutil
import os

import argparse

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='dr classification val level comparing files extraction')
    parser.add_argument('--csvfile', required=True)

    return parser.parse_args()

args = parse_args()

config = args.csvfile

df = pd.DataFrame.from_csv(config)

images_list = []

for index, row in df.iterrows():
    if row[1] == 4 and row[2] <=2:
        images_list.append(row)

df_selected = pd.DataFrame(images_list, columns=['images', 'gt_level', 'pred_level', 'cls_propbality'])

df_selected.to_csv('./select.csv')

print(images_list)
