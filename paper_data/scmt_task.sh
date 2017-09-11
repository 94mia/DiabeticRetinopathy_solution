#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python single_channel_multi_task_cls.py --root ../data/kaggle/512 --traincsv ../data/kaggle/train_multi.csv --valcsv ../data/kaggle/val_multi.csv --testcsv ../data/kaggle/test_bin.csv --model rsn34 --batch 64 --workers 8 --epoch 100 --display 50

#CUDA_VISIBLE_DEVICES=0,1 python single_channel_multi_task_cls.py --root ../data/dme/512 --traincsv ../data/dme/train_multi.csv --valcsv ../data/dme/val_multi.csv --testcsv ../data/dme/test_bin.csv --model rsn34 --batch 64 --workers 8 --epoch 100 --display 50

