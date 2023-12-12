import sys
sys.path.append("../attack_images")
sys.path.append("../utils")

from attack_images import generate_attacks
from utils import utils
from pytorch_ood.detector import MaxSoftmax, ODIN, MaxLogit, Mahalanobis, EnergyBased, KNN, ViM, Entropy
from pytorch_ood.utils import OODMetrics
import numpy as np


def odd_detector(root_path, csv_path, batch_size, image_size, model_path, model_name, attack_name, ood_name, lr, eps):
    
    #1st import dataset
    dataloader_val, num_class = utils.load_attacked_database_df(root_path=root_path, 
                                                                csv_path=csv_path, 
                                                                batch_size=batch_size, 
                                                                image_size=image_size, 
                                                                percentage_attacked=0.1, 
                                                                test_size=0.3)
    
    #2nd load models
    model = utils.read_model_from_checkpoint(model_path=model_path, model_name=model_name, nb_class=num_class)
    model.eval().cuda()
    
    #3rd generate attacked images 
    in_images, out_images, _ = generate_attacks.generate_attack(model=model,
                                                                  attack_name=attack_name, 
                                                                  data_loader=dataloader_val,
                                                                  nb_class=num_class,
                                                                  input_shape=image_size,
                                                                  lr=lr, 
                                                                  eps=eps)
    

    
    in_out_images = np.concatenate((in_images, out_images), axis=0)
    in_out_labels = np.concatenate((np.repeat(0, len(in_images)), np.repeat(-1, len(out_images))), axis=0)
    
    #4th convert images and labels to dataloader
    dataloader_atttacked = utils.numpy_to_dataloader(images=in_out_images, labels=in_out_labels, batch_size=batch_size)
    
    #5th create the detector
    dectetor = __get_ood_strategy(odd_name=ood_name, model=model, eps=eps, t=3.0)
    
    #6th calculate metrics for detector
    metrics = OODMetrics()
    
    for x, y in dataloader_atttacked:
        metrics.update(dectetor(x.cuda()), y)
    
    odd_metrics = dict(metrics.compute())
    
    return odd_metrics


def __get_ood_strategy(odd_name, model, t=1.0, eps=0.01):
    
    ood_strategies = {
        "MaxSoftmax": MaxSoftmax(model=model, t=t),
        "ODIN": ODIN(model=model, eps=eps, temperature=t),
        "MaxLogit": MaxLogit(model=model),
        "Mahalanobis": Mahalanobis(model=model, eps=eps),
        "Energy": EnergyBased(model=model, t=t),
        "KNN": KNN(model=model),
    }
    
    return ood_strategies[odd_name]