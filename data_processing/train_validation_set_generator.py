import argparse

parser = argparse.ArgumentParser(description='generate train & validation set')

parser.add_argument('--root', required=True)
parser.add_argument('--traintxt', required=True)
parser.add_argument('--trainltxt', required=True)

args = parser.parse_args()


files = [f.strip() for f in open(args.traintxt, 'r')]
labels = [int(f.strip()) for f in open(args.trainltxt, 'r')]

cat_files = [[],[],[],[],[]]

print(files)
print(labels)

index = 0
for label in labels:
    cat_files[label].append(files[index])
    index += 1

for index in cat_files:
    print(index)

from sklearn.cross_validation import train_test_split

tr_list = []
te_list = []

for i in range(5):
    tr, te = train_test_split(cat_files[i], test_size=0.1)
    tr_list.append(tr)
    te_list.append(te)


import os
train_dir = os.path.join(args.root, 'train')
val_dir = os.path.join(args.root, 'val')
if not os.path.isdir(val_dir):
    print('create val&train folder!')
    os.mkdir(val_dir)
    os.mkdir(train_dir)

import shutil

train_img_txt = os.path.join(train_dir, 'train_images.txt')
train_label_txt = os.path.join(train_dir, 'train_labels.txt')
val_img_txt = os.path.join(val_dir, 'val_images.txt')
val_label_txt = os.path.join(val_dir, 'val_labels.txt')

f1 = open(train_img_txt, 'w')
f2 = open(train_label_txt, 'w')

f3 = open(val_img_txt, 'w')
f4 = open(val_label_txt, 'w')

for label in range(5):
    tr = tr_list[label]
    te = te_list[label]
    for tr_img in tr:
        raw = os.path.join(args.root, tr_img + '.jpg')
        raw_128 = os.path.join(args.root, tr_img + '_128.png')
        raw_256 = os.path.join(args.root, tr_img + '_256.png')
        raw_512 = os.path.join(args.root, tr_img + '_512.png')
        raw_1024 = os.path.join(args.root, tr_img + '_1024.png')
        raw_list = []
        raw_list.append(raw)
        raw_list.append(raw_128)
        raw_list.append(raw_256)
        raw_list.append(raw_512)
        raw_list.append(raw_1024)
        for r in raw_list:
            if os.path.exists(r):
                shutil.copy(r, train_dir)
        f1.write(tr_img+'\r')
        f2.write(str(label)+'\r')

    for te_img in te:
        raw = os.path.join(args.root, te_img + '.jpg')
        raw_128 = os.path.join(args.root, te_img + '_128.png')
        raw_256 = os.path.join(args.root, te_img + '_256.png')
        raw_512 = os.path.join(args.root, te_img + '_512.png')
        raw_1024 = os.path.join(args.root, te_img + '_1024.png')
        raw_list = []
        raw_list.append(raw)
        raw_list.append(raw_128)
        raw_list.append(raw_256)
        raw_list.append(raw_512)
        raw_list.append(raw_1024)
        if r in raw_list:
            if os.path.exists(r):
                shutil.copy(r, val_dir)
        f3.write(te_img+'\r')
        f4.write(str(label)+'\r')




