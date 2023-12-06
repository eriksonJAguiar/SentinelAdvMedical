import numpy as np
import utils
import torch
from torchmetrics import Accuracy, Recall, Specificity, Precision, F1Score, AUROC

def plot_run_model():
    pass

if __name__ == "__main__":
    
    model_path  = "../clean_train/trained-weights/inceptionv3-melanoma-exp0.ckpt"
    model_name = "inceptionv3"
    attack_img_path  = "../dataset/MelanomaDB"
    folder_from = "DeepFool_inceptionv3" 
    input_size = (229,229)
    batch_size = 32
    
    csv_path = "../dataset/MelanomaDB/ham1000_dataset.csv" 
    
    #1st read database images
    val_data_attack, nb_class = utils.load_attacked_database(path=attack_img_path, folder_from=folder_from, batch_size=batch_size, image_size=input_size)
    
    val_data_clean, nb_class = utils.load_attacked_database_df(root_path="../", csv_path=csv_path, batch_size=batch_size, test_size=0.3)
    
    #2nd read pytorch models
    model = utils.read_model_from_checkpoint(model_path=model_path, model_name=model_name, nb_class=nb_class)
    
    
    val_acc = Accuracy(task="binary") if not nb_class > 2 else Accuracy(task="multiclass", num_classes=nb_class)
    val_auc = AUROC(task="binary") if not nb_class > 2 else AUROC(task="multiclass", num_classes=nb_class, average="macro")
    loss = torch.nn.CrossEntropyLoss()
    val_loss = 0.0
    
    #3rd predict attacked images
    #define loss of the model
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_data_attack):
            x, y = data
            logits = model(x)
            val_loss += loss(logits, y).item()
            
            y_pred = torch.argmax(logits, dim=1) if nb_class > 2 else torch.argmax(logits, dim=1).view(-1, 1).float()
            val_acc(y_pred, y)
            
            print("Val loss: {}".format(val_loss/(i+1)))
            print("Val acc: {}".format(val_acc.compute()))
    
    print("Avg val loss: {}".format(val_loss/(i+1)))
    print("Avg val acc: {}".format(val_acc.compute()))
    
    #4th a 
    
    