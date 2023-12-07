import torch
import numpy as np
import pandas as pd

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, DeepFool, CarliniL2Method, UniversalPerturbation, ProjectedGradientDescent

from torchmetrics import Accuracy, Recall, Specificity, Precision, F1Score, AUROC

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

def evaluate_attack(model, dataset, nb_class):
    pass

def evaluate_model(model, dataset_clean, dataset_adv):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #3rd predict attacked images
    #define loss of the model
    avg_accuracy_clean, avg_accuracy_adv = [], []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataset_clean):
            x, y = data
            x, y = x.to(device), y.to(device)
            pred_clean = model(x)
            y_pred = torch.argmax(pred_clean, dim=1)
            
            accuracy_clean = np.sum(y_pred.cpu().numpy() == y.cpu().numpy()) / len(y)
            avg_accuracy_clean.append(accuracy_clean)
            #print("Accuracy on Clean examples: {}".format(accuracy_clean))
        
        #print("Mean Accuracy on Clean examples: {}\n".format(np.mean(avg_accuracy_clean)))
        
        for i, data in enumerate(dataset_adv):
            x, y = data
            x, y = x.to(device), y.to(device)
            pred_clean = model(x)
            y_pred = torch.argmax(pred_clean, dim=1)
            
            accuracy_adv = np.sum(y_pred.cpu().numpy() == y.cpu().numpy()) / len(y)
            avg_accuracy_adv.append(accuracy_adv)
            #print("Accuracy on Adv examples: {}".format(accuracy_adv))
        
        #print("Mean Accuracy on Adv examples: {}\n".format(np.mean(avg_accuracy_adv)))
        
    epochs_metrics = pd.DataFrame()
    epochs_metrics["epochs"] = list(range(len(dataset_clean)))
    epochs_metrics["val_acc"] = avg_accuracy_clean
    epochs_metrics["val_acc_adv"] = avg_accuracy_adv
    
    # avg_metrics = pd.DataFrame()
    # avg_metrics["clean_metrics"] = np.mean(avg_accuracy_clean)
    # avg_metrics["avg_metrics"] = np.mean(avg_accuracy_adv)
    
    return epochs_metrics
    
    # pred_adv = model(images_adv)
    # accuracy_adv = np.sum(np.argmax(pred_adv, axis=1) == np.argmax(true_labels, axis=1)) / len(true_labels)
    # print("Accuracy on adversarial test examples: {}%".format(accuracy_adv * 100))
    
    # return {"val_acc_clean": accuracy_clean, "val_acc_adv": accuracy_adv}