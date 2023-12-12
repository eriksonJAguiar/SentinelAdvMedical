import sys
sys.path.append("./ood_analysis")
sys.path.append('./attack_images')

from ood_analysis import odd_decting
import torch
import numpy as np


if __name__ == '__main__':
    
    torch.manual_seed(43)
    np.random.seed(43)
    
    dataset_name = "MelanomaDB"
    root_path = "./"
    csv_path = "./dataset/MelanomaDB/ham1000_dataset.csv"
    model_path = "./clean_train/trained-weights/resnet50-MelanomaDB-exp0.ckpt"
    model_name = "resnet50"
    
    metrics = odd_decting.odd_detector(root_path=root_path, 
                                         csv_path=csv_path, 
                                         batch_size=32, 
                                         image_size=(224, 224), 
                                         model_path=model_path, 
                                         model_name=model_name,
                                         attack_name="UAP",
                                         ood_name="MaxSoftmax",
                                         lr=0.0001, 
                                         eps=0.05)
    
    print(metrics)