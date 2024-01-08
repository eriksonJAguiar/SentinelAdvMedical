#!/bin/bash

#for models in "resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
for models in "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
do
    echo "model $models"
    python main.py --dataset ../ --dataset_csv ../dataset/PnemoniaDB/pneumonia_db.csv --model_name $models --as_augmentation --epochs 50 --dataset_name pneumonia
done