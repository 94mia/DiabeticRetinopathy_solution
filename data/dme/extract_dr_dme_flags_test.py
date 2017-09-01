import os
import pandas as pd
import numpy as np
import argparse
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description='extract dr&dme flags')
    parser.add_argument('--root', required=True)
    parser.add_argument('--csvfile', required=True)

    return parser.parse_args()

args = parse_args()

print('====> args:')
print(args)

dme_path = os.path.join(args.root, 'dme')
assert os.path.isdir(dme_path)

csv_file = os.path.join(args.root, args.csvfile)

images_list = glob(os.path.join(dme_path, '*.jpg'))

images_list = [os.path.basename(img) for img in images_list]

# dr_level_list = [int(index.split('.')[0].split('_')[2]) for index in images_list]
# dme_level_list = [int(index.split('.')[0].split('_')[4]) for index in images_list]

print('raw images count is: {}'.format(len(images_list)))

images_list_filtered = []
dr_list_filtered = []
dme_list_filtered = []

for img in images_list:
    index = img.split('.')[0]
    dr_level = int(index.split('_')[2])
    dme_level = int(index.split('_')[4])
    if dr_level <= 1  and dme_level > 0:
        continue
    images_list_filtered.append(index)
    dr_list_filtered.append(dr_level)
    dme_list_filtered.append(dme_level)

assert len(images_list_filtered) == len(dr_list_filtered) == len(dme_list_filtered)

print('filtered images count is: {}'.format(len(images_list_filtered)))

data = np.column_stack((images_list_filtered, dr_list_filtered, dme_list_filtered))

df = pd.DataFrame(data, columns=['image', 'dr_level', 'dme_level'])

df.to_csv(csv_file)
