import attack_images.utils as utils
import generate_attacks
import torch
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('-dm','--dataset_name', help='databaset name')
parser.add_argument('-d','--dataset', help='databaset path', required=False)
parser.add_argument('-dv','--dataset_csv', help='databaset csv file', required=False)

parser.add_argument('-mn', '--model_name', help="model to training name: resnet50 or resnet18", required=True)
parser.add_argument('-wp', '--weights_path', help="model weigths path", required=True)

parser.add_argument('-an', '--attack_name', help="Attack name FGSM, PGD, CW or UAP", required=True)
parser.add_argument('-e', '--eps', help="Attack noise", required=True)
parser.add_argument('-pa', '--path_attack', help="Attack noise", required=True)

args = vars(parser.parse_args())

if __name__ == '__main__':

    torch.manual_seed(123)
    np.random.seed(123)
    
    dataset_name = args["dataset_name"]
    root_path = args["dataset"]
    csv_path = args["dataset_csv"]
    model_name = args["model_name"]
    input_size = (299, 299) if model_name == "inceptionv3" else (224, 224)
    batch_size = 32
    lr = 0.0001
    
    print("Load validation database using {}...".format(dataset_name))
    val_attack_dataset, num_class = utils.load_attacked_database_df(root_path=root_path, csv_path=csv_path, batch_size=batch_size, image_size=input_size, percentage_attacked=0.2, test_size=0.3)
    
    model_path = args["weights_path"]
    attack_name = args["attack_name"]
    eps = float(args["eps"])
    path_attack_to = args["path_attack"]
    
    model = utils.read_model_from_checkpoint(model_path=model_path, model_name=model_name, nb_class=num_class)
    
    print("Generate attacked images using attack {}...".format(attack_name))
    print("The eps is {}".format(str(eps)))
    adv_images, true_labels = generate_attacks.generate_attack(
                        model=model,
                        input_shape=input_size,
                        lr=lr,
                        nb_class=num_class,
                        attack_name=attack_name,
                        data_loader=val_attack_dataset,
                        eps=eps
                    )
    
    #print("save attacked images...")
    #utils.save_all_adv_image(path_to_save=path_attack_to, images_array=adv_images, labels=true_labels, db_name=dataset_name, attack_name="{}_{}".format(attack_name, model_name))
    
    
    