import torch
import pandas as pd
import numpy as np

from torchmetrics import Accuracy, Recall, Specificity, Precision, F1Score, AUROC

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
        
        for i, data in enumerate(dataset_adv):
            x, y = data
            x, y = x.to(device), y.to(device)
            pred_clean = model(x)
            y_pred = torch.argmax(pred_clean, dim=1)
            
            accuracy_adv = np.sum(y_pred.cpu().numpy() == y.cpu().numpy()) / len(y)
            avg_accuracy_adv.append(accuracy_adv)
        
    epochs_metrics = pd.DataFrame()
    epochs_metrics["epochs"] = list(range(len(dataset_clean)))
    epochs_metrics["val_acc"] = avg_accuracy_clean
    epochs_metrics["val_acc_adv"] = avg_accuracy_adv
    
    return epochs_metrics