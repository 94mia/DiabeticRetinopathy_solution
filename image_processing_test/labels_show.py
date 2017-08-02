import argparse
import os

parser = argparse.ArgumentParser(description='generate train & validation set')

parser.add_argument('--root', required=True)
parser.add_argument('--testratio', type=float, default=0.1)

args = parser.parse_args()

root = args.root

images_path = os.path.join(root, 'train_images.txt')
labels_path = os.path.join(root, 'train_labels.txt')

f1 = open(images_path, 'r')

images_list = [line.strip() for line in f1]

list1 = list(set(images_list))

print(len(images_list))
print(len(list1))