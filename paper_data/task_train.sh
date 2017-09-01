CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_task_classification.py --root ../data/dme/512 --traincsv ../data/dme/train.csv --valcsv ../data/dme/val.csv --testcsv ../data/dme/test.csv --model rsn18 --batch 200 --workers 8
CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_task_classification.py --root ../data/dme/512 --traincsv ../data/dme/train.csv --valcsv ../data/dme/val.csv --testcsv ../data/dme/test.csv --model rsn34 --batch 100 --workers 8
CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_task_classification.py --root ../data/dme/512 --traincsv ../data/dme/train.csv --valcsv ../data/dme/val.csv --testcsv ../data/dme/test.csv --model rsn50 --batch 64 --workers 8
CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_task_classification.py --root ../data/dme/512 --traincsv ../data/dme/train.csv --valcsv ../data/dme/val.csv --testcsv ../data/dme/test.csv --model rsn101 --batch 32 --workers 8
CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_task_classification.py --root ../data/dme/512 --traincsv ../data/dme/train.csv --valcsv ../data/dme/val.csv --testcsv ../data/dme/test.csv --model dsn121 --batch 32 --workers 8


CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_channel_classification.py --root ../data/dme/512 --root_augumentation ../data/dme/512_dme --traincsv ../data/dme/train.csv --valcsv ../data/dme/val.csv --testcsv ../data/dme/test.csv --model rsn18 --batch 100 --workers 8
CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_channel_classification.py --root ../data/dme/512 --root_augumentation ../data/dme/512_dme --traincsv ../data/dme/train.csv --valcsv ../data/dme/val.csv --testcsv ../data/dme/test.csv --model rsn34 --batch 64 --workers 8
CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_channel_classification.py --root ../data/dme/512 --root_augumentation ../data/dme/512_dme --traincsv ../data/dme/train.csv --valcsv ../data/dme/val.csv --testcsv ../data/dme/test.csv --model rsn50 --batch 32 --workers 8
CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_channel_classification.py --root ../data/dme/512 --root_augumentation ../data/dme/512_dme --traincsv ../data/dme/train.csv --valcsv ../data/dme/val.csv --testcsv ../data/dme/test.csv --model rsn101 --batch 16 --workers 8
CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_channel_classification.py --root ../data/dme/512 --root_augumentation ../data/dme/512_dme --traincsv ../data/dme/train.csv --valcsv ../data/dme/val.csv --testcsv ../data/dme/test.csv --model dsn121 --batch 16 --workers 8


