#!/bin/bash

for models in "resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
do
    echo "model $models"
    python3 main.py --dataset ../dataset/OCT/  --model_name $models --exp_num 1 --epochs 50 --dataset_name oct 
done 