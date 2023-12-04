#!/bin/bash

for models in "resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
do
    echo "model $models"
    for attacks in "FGSM" "PGD" "UAP" "DeepFool" "CW"
    do
        echo "attack $attacks"
        python3 main.py --dataset ../ --dataset_csv ../dataset/MelanomaDB/ham1000_dataset.csv  --dataset_name MelanomaDB --model_name $models --weights_path "../clean-train/trained-weights/$models-melanoma-exp0.ckpt" --attack_name $attacks --eps 0.01 --path_attack ../dataset
    done
done 