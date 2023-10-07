from utils import UtilsTroch, DatasetMedmnistCustom
from train_model import PytorchTrainingAndTest
from models import ModelsPretrained
import argparse
import pandas as pd
import torch
import numpy as np
import os

#Update CUDA version
#pip install --user torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Device to run: {device}")

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='')
parser.add_argument('-d','--dataset', help='databaset path', required=False)
parser.add_argument('-dm','--dataset_name', help='databaset name', required=False)
parser.add_argument('-dv','--dataset_csv', help='databaset csv file', required=False)
parser.add_argument('-mn', '--medmnist_db', required=False)
parser.add_argument('-m', '--model_name', help="model to training name: resnet50 or resnet18", required=True)
parser.add_argument('-r', '--as_rgb', action="store_true", required=False)
parser.add_argument('-au', '--as_augmentation', action="store_true", required=False)
parser.add_argument("-t", "--test_size", required=False)
parser.add_argument("-e", "--exp_num")
parser.add_argument("-ep", "--epochs")
args = vars(parser.parse_args())
    
if __name__ == '__main__':

    torch.manual_seed(123)
    np.random.seed(123)
    
    train_with_pytorch = PytorchTrainingAndTest()
    models_load = ModelsPretrained()
    
    #Variables to run experiments
    learning_rate = 0.0001
    num_epochs = int(args["epochs"])
    batch_size = 32
    test_size = None if args["test_size"] is None else float(args["test_size"])
    number_experiments = int(args["exp_num"])
    as_aug = args["as_augmentation"]
    model_name = args["model_name"]
    #image_size = (224, 224)
    image_size = (299, 299) if model_name == "inceptionv3" else (224, 224)

    medmnist = args["medmnist_db"]
    if medmnist is None:
        base_path = args["dataset"]  
        csv_path = args["dataset_csv"]
        database_name = args["dataset_name"]
    else:
        medmnist = str(args["medmnist_db"]).lower()
        database_name = medmnist
        as_rgb = args["as_rgb"]
        base_path = None
        csv_path = None
    
      
    print("Database: {}".format(database_name))
            

    results_metrics = pd.DataFrame()
    for exp_num in range(number_experiments):
        print(f"Starting experiment {exp_num+1}")
        print("Loading database ...")
        if medmnist:
            train, test, num_class, task = UtilsTroch.load_medmnist(database_name=medmnist, image_size=image_size, batch_size=batch_size, as_rgb=as_rgb, balanced=False)
        else:
            #train, test, num_class = UtilsTroch.load_database_df(root_path=base_path, batch_size=batch_size, image_size=image_size,csv_path=csv_path, is_agumentation=as_aug, test_size=test_size)
            #train, test, num_class = UtilsTroch.load_database_kf(path_image=base_path, batch_size=batch_size, image_size=image_size,  n_folds=5, csv_path=csv_path)
            train, test, num_class = UtilsTroch.load_database(path=base_path, batch_size=batch_size, image_size=image_size, is_agumentation=as_aug)
            UtilsTroch.show_images(train, database_name)

        print(f"Number of class: {str(num_class)}")
        
        #models selected
        # arquitetures_pretrained = {
        #                         "resnet50" : models_load.make_model_pretrained("resnet50", num_class),
        #                         #"resnet18" : models_load.make_model_pretrained("resnet18", num_class),
        #                         "densenet" : models_load.make_model_pretrained("densenet", num_class),
        #                         "vgg16" : models_load.make_model_pretrained("vgg16", num_class),
        #                         "vgg19" : models_load.make_model_pretrained("vgg19", num_class),
        #                         #"efficientnet" : models_load.make_model_pretrained("efficientnet", num_class),
        #                         "inceptionv3" : models_load.make_model_pretrained("inceptionv3", num_class),
        #                     }
        
        model_config = models_load.make_model_pretrained(model_name, num_class)
        
        #train_model = {}
        #test_model = {}
        #for model_name, model_config in arquitetures_pretrained.items():
            #model_select = models_load.make_model_pretrained(model_name, num_class)
        print("\nNetwork: "+ model_name + " is training...\n")
        results = train_with_pytorch.run_model(exp_num=exp_num,model=model_config, model_name=model_name, 
                                                                database_name=database_name, train=train, test=test, 
                                                                learning_rate=learning_rate, num_epochs=num_epochs, num_class=num_class)
                # train_with_pytorch.run_model(model=model_select, model_name=model_name, database_name=database_name, train=train, test=test, 
                #                              learning_rate=learning_rate, num_epochs=num_epochs, num_class=num_class, batch_size=batch_size, 
                #                            image_size=image_size)
                #train_model[model_name] = 
                
        results_metrics = pd.concat([results_metrics, results])
            
    print(results_metrics)
    if os.path.exists("../metrics/results_{}.csv".format(database_name)):
        results_metrics.to_csv("../metrics/results_{}.csv".format(database_name), mode="a", header=None)
    else:
        results_metrics.to_csv("../metrics/results_{}.csv".format(database_name))
        
        