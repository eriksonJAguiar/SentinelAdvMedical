#!/bin/bash

for models in "resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
do
    echo "model $models"
    python3 main.py --dataset ../ --dataset_csv ../dataset/MelanomaDB/ham1000_dataset.csv --model_name $models --as_augmentation --epochs 50 --dataset_name melanoma --exp_num 1 --test_size 0.3
done 