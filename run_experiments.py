import sys
sys.path.append('./attack_images')

from attack_images import generate_attacks
import torch
import numpy as np
import argparse


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

    torch.manual_seed(123)
    np.random.seed(123)
    
    #1st define de args
    dataset_name = args["dataset_name"]
    root_path = args["dataset"]
    csv_path = args["dataset_csv"]
    weights_path = args["weights_path"]
    
    #2nd define parameters
    batch_size = 32
    lr = 0.0001
    models = ["resnet50", "vgg16", "vgg19", "inceptionv3", "efficientnet", "densenet"]
    attacks = ["FGSM", "PGD", "UAP", "DeepFool", "CW"] #["FGSM", "PGD", "UAP"]
    epsilons = [0.001, 0.01, 0.05, 0.1, 0.5]
    
    for model_name in models:
        print("Starting attack for model {}...".format(model_name))
        input_size = (299, 299) if model_name == "inceptionv3" else (224, 224)
        for attack_name in attacks:
            print("Generate attacked images using attack {}...".format(attack_name))
            for eps in epsilons: 
                print("The eps is {}".format(str(eps)))
                #5th run attack
                generate_attacks.run_attack(root_path=root_path, 
                                               dataset_name=dataset_name, 
                                               csv_path=csv_path, 
                                               weights_path=weights_path, 
                                               model_name=model_name,
                                               input_size=input_size,
                                               attack_name=attack_name, 
                                               eps=eps, 
                                               batch_size=batch_size, 
                                               lr=lr,
                                               save_metrics_path="./metrics")
    
    