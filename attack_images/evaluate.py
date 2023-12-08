import torch
import pandas as pd
import numpy as np

from torchmetrics import Accuracy, Recall, Specificity, Precision, F1Score, AUROC
from sklearn.metrics import roc_auc_score

def evaluate_model(model, dataset_clean, dataset_adv, nb_class):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #3rd predict attacked images
    #define loss of the model
    avg_accuracy_clean, avg_accuracy_adv = [], []
    avg_auc_clean, avg_auc_adv = [], []
    asr = []
    
    adv_auc = AUROC(task="binary") if not nb_class > 2 else AUROC(task="multiclass", num_classes=nb_class, average="macro")
    val_auc = AUROC(task="binary") if not nb_class > 2 else AUROC(task="multiclass", num_classes=nb_class, average="macro")
    
    model.eval()
    with torch.no_grad():
        for i, (data_clean, data_adv) in enumerate(zip(dataset_clean, dataset_adv)):
            #test images for clean examples
            x_clean, y_clean = data_clean
            x_clean, y_clean = x_clean.to(device), y_clean.to(device)
            pred_clean = model(x_clean)
            y_clean = y_clean if nb_class > 2 else y_clean.view(-1, 1).float()
            y_pred = torch.argmax(pred_clean, dim=1) if nb_class > 2 else torch.argmax(pred_clean, dim=1).view(-1, 1).float()
            y_prob = torch.softmax(pred_clean, dim=1) if nb_class > 2 else torch.sigmoid(pred_clean)
            
            accuracy_clean = np.sum(y_pred.cpu().numpy() == y_clean.cpu().numpy()) / len(y_clean)
            avg_accuracy_clean.append(accuracy_clean)
            val_auc(y_prob, y_clean)
            
            avg_auc_clean.append(val_auc.compute().cpu().numpy())
        
            #test images for adversarial examples
            x_adv, y_adv = data_adv
            x_adv, y_adv = x_adv.to(device), y_adv.to(device)
            pred_adv = model(x_adv)
            y_adv = y_adv if nb_class > 2 else y_adv.view(-1, 1).float()
            y_pred = torch.argmax(pred_adv, dim=1) if nb_class > 2 else torch.argmax(pred_adv, dim=1).view(-1, 1).float()
            y_prob = torch.softmax(pred_adv, dim=1) if nb_class > 2 else torch.sigmoid(pred_adv)
 
            accuracy_adv = np.sum(y_pred.cpu().numpy() == y_adv.cpu().numpy()) / len(y_adv)
            avg_accuracy_adv.append(accuracy_adv)
            adv_auc(y_prob, y_adv)
            
            avg_auc_adv.append(adv_auc.compute().cpu().numpy())
            #evaluate the attack sucess rate (asr)
            asr.append(1 - accuracy_adv)
        
    epochs_metrics = pd.DataFrame()
    epochs_metrics["epochs"] = list(range(len(dataset_clean)))
    epochs_metrics["val_acc"] = avg_accuracy_clean
    epochs_metrics["val_acc_adv"] = avg_accuracy_adv
    epochs_metrics["val_auc"] = avg_auc_clean
    epochs_metrics["val_auc_adv"] = avg_auc_adv
    epochs_metrics["asr"] = asr
    
    return epochs_metrics