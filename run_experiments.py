import sys
sys.path.append('./attack_images')
sys.path.append("./ood_analysis")
sys.path.append("../utils")

from ood_analysis import odd_decting
from attack_images import generate_attacks
from utils import utils
import torch
import numpy as np
import argparse
import pandas as pd
import os
import time
import gc


parser = argparse.ArgumentParser(description='')
parser.add_argument('-dm','--dataset_name', help='databaset name')
parser.add_argument('-d','--dataset', help='databaset path', required=False)
parser.add_argument('-dv','--dataset_csv', help='databaset csv file', required=False)

# parser.add_argument('-mn', '--model_name', help="model to training name: resnet50 or resnet18", required=True)
parser.add_argument('-wp', '--weights_path', help="root of model weigths path", required=True)

# parser.add_argument('-an', '--attack_name', help="Attack name FGSM, PGD, CW or UAP", required=True)
# parser.add_argument('-e', '--eps', help="Attack noise", required=True)
#parser.add_argument('-pa', '--path_attack', help="Attack noise", required=True)

args = vars(parser.parse_args())

if __name__ == '__main__':

    torch.manual_seed(43)
    np.random.seed(43)
    
    #1st define de args
    dataset_name = args["dataset_name"]
    root_path = args["dataset"]
    csv_path = args["dataset_csv"]
    weights_path = args["weights_path"]
    
    #2nd define parameters
    nb_class = 7
    batch_size = 32
    lr = 0.001
    models = ["resnet50", "vgg16","vgg19","inceptionv3", "efficientnet", "densenet"]
    attacks = ["FGSM", "BIM", "PGD", "DeepFool", "UAP", "CW"] 
    epsilons = [0.001, 0.01, 0.05, 0.1, 0.5]
    ood_strategy = ["MaxSoftmax","ODIN", "MaxLogit", "Entropy", "Mahalanobis", "MCD"]
    
    for model_name in models:
        print("Starting attack for model {}...".format(model_name))
        input_size = (299, 299) if model_name == "inceptionv3" else (224, 224)
        for attack_name in attacks:
            print("Generate attacked images using attack {}...".format(attack_name))
            for eps in epsilons: 
                print("The eps is {}".format(str(eps)))
                #5th run attack
                images, adv_images, true_labels = generate_attacks.run_attack(root_path=root_path, 
                                                                              dataset_name=dataset_name, 
                                                                              csv_path=csv_path, 
                                                                              weights_path=weights_path, 
                                                                              model_name=model_name,
                                                                              input_size=input_size,
                                                                              attack_name=attack_name, 
                                                                              eps=eps, 
                                                                              batch_size=batch_size, 
                                                                              lr=lr,
                                                                              save_metrics_path="./metrics",
                                                                              is_logits_save=True,
                                                                              is_features_save=True)
                
                utils.show_random_adv_image(adv_images[:32], dataset_name, attack_name, eps,path_to_save="./metrics/figures/attacks")
                
                for ood in ood_strategy:
                    time_odd = time.time()
                    metrics_ood = odd_decting.odd_detector2(weights_path=weights_path,
                                                        dataset_name=dataset_name,
                                                        model_name=model_name,
                                                        in_images=images, 
                                                        out_images=adv_images, 
                                                        batch_size=batch_size,
                                                        ood_name=ood,
                                                        eps=eps,
                                                        nb_class=nb_class)
                    end_ood = time.time() - time_odd
                    
                    metrics_ood["OOD"] = ood
                    metrics_ood["model"] = model_name
                    metrics_ood["attack"] = attack_name
                    metrics_ood["dataset"] = dataset_name
                    metrics_ood["eps"] = eps
                    metrics_ood["time"] = end_ood
                    metrics_ood = {k:[v] for k,v in metrics_ood.items()}
                    
                    if os.path.exists("./metrics/metrics_ood.csv"):
                        pd.DataFrame.from_dict(metrics_ood).to_csv("./metrics/metrics_ood.csv", index=False, header=False, mode="a")
                    else:
                        pd.DataFrame.from_dict(metrics_ood).to_csv("./metrics/metrics_ood.csv", index=False, header=True, mode="a")
                    
                    #clear cuda memory
                    torch.cuda.empty_cache()
                    gc.collect()        
    
    