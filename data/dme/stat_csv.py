import pandas as pd

import argparse

import os

def parse_args():
    parser = argparse.ArgumentParser(description='stat csv')
    parser.add_argument('--root', default='./')

    parser.add_argument('--csvdata', default='data_multi.csv')
    parser.add_argument('--csvtrain', default='train_multi.csv')
    parser.add_argument('--csvval', default='val_multi.csv')
    parser.add_argument('--csvtest', default='test_multi.csv')

    return parser.parse_args()

args = parse_args()

csv_train = os.path.join(args.root, args.csvtrain)
csv_val = os.path.join(args.root, args.csvval)
csv_test = os.path.join(args.root, args.csvtest)


csv = csv_train

df = pd.DataFrame.from_csv(csv)

images_list = []
for index, row in df.iterrows():
    images_list.append(row)

cnt0 = 0
cnt1 = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0


for i in range(len(images_list)):
    dr = images_list[i][1]
    if dr == 0:
        cnt0 +=1
    elif dr == 1:
        cnt1 += 1
    elif dr == 2:
        cnt2 += 1
    elif dr == 3:
        cnt3 += 1
    elif dr == 4:
        cnt4 += 1

cnt_sum = cnt0 + cnt1 + cnt2 + cnt3 + cnt4

assert cnt_sum == len(images_list)

print('dr cnt0: {:.4f}'.format(cnt0/cnt_sum))
print('dr cnt1: {:.4f}'.format(cnt1/cnt_sum))
print('dr cnt2: {:.4f}'.format(cnt2/cnt_sum))
print('dr cnt3: {:.4f}'.format(cnt3/cnt_sum))
print('dr cnt4: {:.4f}'.format(cnt4/cnt_sum))


cnt0 = 0
cnt1 = 0
cnt2 = 0
cnt3 = 0

for i in range(len(images_list)):
    dr = images_list[i][2]
    if dr == 0:
        cnt0 +=1
    elif dr == 1:
        cnt1 += 1
    elif dr == 2:
        cnt2 += 1
    elif dr == 3:
        cnt3 += 1

cnt_sum = cnt0 + cnt1 + cnt2 + cnt3

assert cnt_sum == len(images_list)

print('\ndme cnt0: {:.4f}'.format(cnt0/cnt_sum))
print('dme cnt1: {:.4f}'.format(cnt1/cnt_sum))
print('dme cnt2: {:.4f}'.format(cnt2/cnt_sum))
print('dme cnt3: {:.4f}'.format(cnt3/cnt_sum))


cnt0 = 0
cnt1 = 0

for i in range(len(images_list)):
    dr = images_list[i][3]
    if dr == 0:
        cnt0 +=1
    elif dr == 1:
        cnt1 += 1

cnt_sum = cnt0 + cnt1

assert cnt_sum == len(images_list)

print('\nreferable cnt0: {:.4f}'.format(cnt0/cnt_sum))
print('referable cnt1: {:.4f}'.format(cnt1/cnt_sum))