#!/bin/bash

#for models in "resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
for models in "resnet50"
do
    echo "model $models"
    python main.py --dataset ../ --dataset_csv ../dataset/nhi_chest_xray/nhi_dataset_chest8.csv --model_name $models --as_augmentation --epochs 50 --as_per_class --dataset_name NHI_perclass
done