

CUDA_VISIBLE_DEVICES=0,1,2,3 python single_channel_multi_task_cls_dme.py --root ../data/zhizhen_new/LabelImages/512 --traincsv ../data/zhizhen_new/LabelImages/train_multi.csv --valcsv ../data/zhizhen_new/LabelImages/val_multi.csv --testcsv ../data/zhizhen_new/LabelImages/test_multi.csv --model rsn34 --batch 64 --workers 4 --epoch 50 --display 50 --crop 448 --size 512 --exp single_net_dme --dme_weight_aug 8.0

CUDA_VISIBLE_DEVICES=0,1,2,3 python single_channel_multi_task_cls_dr.py --root ../data/zhizhen_new/LabelImages/512 --traincsv ../data/zhizhen_new/LabelImages/train_multi.csv --valcsv ../data/zhizhen_new/LabelImages/val_multi.csv --testcsv ../data/zhizhen_new/LabelImages/test_multi.csv --model rsn34 --batch 64 --workers 4 --epoch 50 --display 50 --crop 448 --size 512 --exp single_net_dr

