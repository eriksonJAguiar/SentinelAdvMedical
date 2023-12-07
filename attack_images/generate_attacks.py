import torch
import numpy as np
import pandas as pd
import utils
import os
import time
import evaluate

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, DeepFool, CarliniL2Method, UniversalPerturbation, ProjectedGradientDescent

#import foolbox as fb
#from foolbox.attacks import L2FastGradientAttack, L2CarliniWagnerAttack, L2DeepFoolAttack

def generate_attack(model, data_loader, input_shape, lr, nb_class, attack_name, eps):
    
    #1st: read a pytorch model
    # checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    # state_dict = {key[6:] : checkpoint['state_dict'][key] for key in checkpoint['state_dict']}
    # model = __get_model_structure(model_name, nb_class)
    # model.load_state_dict(state_dict)
    # #model = __get_model_last_layers(model_name, model, nb_class)
    # model.eval()
    
    #2nd define the loss and optimizer
    loss = torch.nn.CrossEntropyLoss() if nb_class > 2 else torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    #3rd create ART classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=opt,
        #clip_values=[0,1],
        input_shape=input_shape,
        nb_classes=nb_class
    )
    
    #4th execute the attack type
    images, adv_images, true_labels = __get_adv_attack(attack_name=attack_name, 
                                               data_loader=data_loader, 
                                               nb_class=nb_class, 
                                               classifier=classifier, 
                                               eps=eps)
    
    return images, adv_images, true_labels

def __get_adv_attack(attack_name, data_loader, nb_class, classifier, eps):
    
    attack = None
    
    #load images and labels
    images, labels = zip(*[data_loader.dataset[i] for i in range(len(data_loader.dataset))])
    images = torch.stack(images).numpy() 
    true_labels = np.array(labels)
    
    #FGSM, DeepFool, C&W, UAP
    if attack_name == "FGSM":
        attack = FastGradientMethod(estimator=classifier, eps=eps, batch_size=32)
    elif attack_name == "DeepFool":
        attack = DeepFool(classifier=classifier, epsilon=eps, batch_size=32, max_iter=5)
    elif attack_name == "CW":
         attack = CarliniL2Method(classifier=classifier, max_iter=5, batch_size=32)
        #true_labels = __get_one_hot(true_labels, nb_class)
    elif attack_name == "PGD":
        attack = ProjectedGradientDescent(estimator=classifier, eps=eps, batch_size=32)    
    elif attack_name == "UAP":
        attack = UniversalPerturbation(classifier=classifier, attacker="pgd", eps=eps, max_iter=5, batch_size=32)
    
    adv_attack = attack.generate(x=images)
    
    return images, adv_attack, true_labels

def run_attack(root_path, dataset_name, csv_path, weights_path, model_name, attack_name, eps, batch_size, lr, save_metrics_path):    
   
    input_size = (299, 299) if model_name == "inceptionv3" else (224, 224)
        
    print("Load validation database using {}...".format(dataset_name))
    #3rd read validation dataset to attack the model
    val_attack_dataset, num_class = utils.load_attacked_database_df(root_path=root_path, csv_path=csv_path, batch_size=batch_size, image_size=input_size, percentage_attacked=0.2, test_size=0.3)
        
    #4th read models from checkpoints
    model_path = os.path.join(weights_path, "{}-{}-exp0.ckpt".format(model_name, dataset_name))
    model = utils.read_model_from_checkpoint(model_path=model_path, model_name=model_name, nb_class=num_class)
        
    print("Generate attacked images using attack {}...".format(attack_name))
    print("The eps is {}".format(str(eps)))
    
    #5th run attack
    time_start = time.time()
    images, adv_images, true_labels = generate_attack(
                            model=model,
                            input_shape=input_size,
                            lr=lr,
                            nb_class=num_class,
                            attack_name=attack_name,
                            data_loader=val_attack_dataset,
                            eps=eps
                        )
    final_time = time.time() - time_start
                
    #6th convert images and labels to dataloader
    loader_clean = utils.numpy_to_dataloader(images=images, labels=true_labels, batch_size=32)
    loader_adv = utils.numpy_to_dataloader(images=adv_images, labels=true_labels, batch_size=32)
                
    #7th evaluate accuracy of the models
    metrics_epochs = evaluate.evaluate_model(model=model, dataset_clean=loader_clean, dataset_adv=loader_adv)
    size = len(metrics_epochs["epochs"])
                
    #8th define metrics
    metrics_avg = pd.DataFrame([{"model": model_name, "attack": attack_name, "eps": eps, "val_acc": metrics_epochs["val_acc"].mean(), "val_acc_adv": metrics_epochs["val_acc_adv"].mean()}])
    metrics_time = pd.DataFrame([{"attack": attack_name, "examples": len(images),"time": final_time}])
    metrics_epochs["model"] = np.repeat(model_name, size)
    metrics_epochs["attack"] = np.repeat(attack_name, size)
    metrics_epochs["eps"] = np.repeat(eps, size)
                
    #9th save metrics to CSV
    avg_path = os.path.join(save_metrics_path, "attacks_avg.csv")
    epochs_path = os.path.join(save_metrics_path, "attacks_epochs.csv")
    time_path = os.path.join(save_metrics_path, "attacks_epochs.csv")
    
    metrics_avg.to_csv(avg_path, index=False, mode="a", header=False if os.path.exists(avg_path) else True)
    metrics_epochs.to_csv(epochs_path, index=False, mode="a", header=False if os.path.exists(epochs_path) else True)
    metrics_time.to_csv(time_path, index=False, mode="a", header=False if os.path.exists(time_path) else True)
    