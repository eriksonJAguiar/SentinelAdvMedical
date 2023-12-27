#!/bin/bash

#for models in "resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
for models in "resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
do
    echo "model $models"
    python main.py --dataset ../ --dataset_csv ../dataset/MelanomaDB/ham1000_dataset_bkp.csv --model_name $models --as_augmentation --as_rgb --kfold 5 --epochs 50 --dataset_name MelanomaDB
done