#!/usr/bin/env bash

#python zhizhen_data.py

# generate 512 size data
#python ../../data_processing/data_preprocessing_scale.py --root ./LabelImages/abnormal --workers 4

#mkdir ./LabelImages/512

mv ./LabelImages/abnormal/512 ./LabelImages/

# generate ahe data
python ../../paper_data/dr_data_ahe.py --root ./LabelImages --workers 16

zip -r 512.zip ./LabelImages/512
zip -r 512_ahe.zip ./LabelImages/512_ahe



