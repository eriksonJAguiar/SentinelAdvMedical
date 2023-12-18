#!/bin/bash

for models in "resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
do
    echo "model $models"
    python main.py --dataset ../ --dataset_csv ../dataset/MelanomaDB/ham1000_dataset.csv --model_name $models --as_augmentation --as_rgb --epochs 50 --dataset_name MelanomaDB --exp_num 1
done