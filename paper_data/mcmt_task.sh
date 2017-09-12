#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_channel_multi_task_classification.py --root ../data/zhizhen_new/LabelImages/512 --root_augumentation ../data/zhizhen_new/LabelImages/512_ahe --traincsv ../data/zhizhen_new/LabelImages/val_multi.csv --valcsv ../data/zhizhen_new/LabelImages/val_multi.csv --testcsv ../data/zhizhen_new/LabelImages/test_bin.csv --model rsn18 --batch 32 --workers 8 --epoch 150 --display 20

#CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_channel_multi_task_classification.py --root ../data/zhizhen_new/LabelImages/512 --root_augumentation ../data/zhizhen_new/LabelImages/512_ahe --traincsv ../data/zhizhen_new/LabelImages/train_multi.csv --valcsv ../data/zhizhen_new/LabelImages/val_multi.csv --testcsv ../data/zhizhen_new/LabelImages/test_bin.csv --model rsn34 --batch 32 --workers 8 --epoch 100 --display 50
#
#CUDA_VISIBLE_DEVICES=0,/1,2,3 python multi_channel_multi_task_classification.py --root ../data/zhizhen_new/LabelImages/512 --root_augumentation ../data/zhizhen_new/LabelImages/512_ahe --traincsv ../data/zhizhen_new/LabelImages/train_multi.csv --valcsv ../data/zhizhen_new/LabelImages/val_multi.csv --testcsv ../data/zhizhen_new/LabelImages/test_bin.csv --model dsn121 --batch 8 --workers 4 --epoch 100 --display 200
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_channel_multi_task_classification.py --root ../data/zhizhen_new/LabelImages/512 --root_augumentation ../data/zhizhen_new/LabelImages/512_ahe --traincsv ../data/zhizhen_new/LabelImages/train_multi.csv --valcsv ../data/zhizhen_new/LabelImages/val_multi.csv --testcsv ../data/zhizhen_new/LabelImages/test_bin.csv --model rsn101 --batch 8 --workers 4 --epoch 100 --display 200


CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_channel_multi_task_classification.py --root ../data/zhizhen_new/LabelImages/512 --root_augumentation ../data/zhizhen_new/LabelImages/512_ahe --traincsv ../data/zhizhen_new/LabelImages/train_multi.csv --valcsv ../data/zhizhen_new/LabelImages/val_multi.csv --testcsv ../data/zhizhen_new/LabelImages/test_bin.csv --model rsn34 --batch 32 --workers 8 --epoch 100 --display 50
