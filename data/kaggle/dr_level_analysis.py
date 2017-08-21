import os
from glob import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='dr level analysis')
parser.add_argument('--root', required=True)
parser.add_argument('--imagestxt', default='train_images.txt')
parser.add_argument('--labelstxt', default='train_labels.txt')
args = parser.parse_args()

root = args.root
imagestxt = args.imagestxt
labelstxt = args.labelstxt

images_list = [line.strip() for line in open(os.path.join(root, imagestxt))]
labels_list = [line.strip() for line in open(os.path.join(root, labelstxt))]

assert len(images_list) == len(labels_list)

print('images list length is: {}'.format(len(images_list)))

dict = {}

for index, imageid in enumerate(images_list):
    patientid = imageid.split('_')[0]
    eye = imageid.split('_')[1]
    if not patientid in dict:
        list = []
        list.append(0)
        list.append(0)
        dict[patientid] = list
    list = dict[patientid]
    if eye == 'left':
        list[0] = int(labels_list[index])
    elif eye == 'right':
        list[1] = int(labels_list[index])
    dict[patientid] = list

assert len(images_list)/2 == len(dict)

print('patient count is: {}'.format(len(dict)))

eyes = np.zeros((5,5), dtype=int)

eyes[0][0] += 1

print(eyes)

for key,value in dict.items():
    eyes[value[0]][value[1]] += 1
    print('{0}  {1}'.format(value[0], value[1]))

print(eyes)

# for index, imageid in enumerate(images_list):
#     if 'left' in imageid and labels_list[index] != '0':
#         print(imageid+': '+labels_list[index])