
from torch import save, utils as thutils
from torchvision import transforms, datasets, utils
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

RANDOM_SEED = 123 

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_attacked_database(path, batch_size, image_size=(128,128), is_agumentation=False):
        tf_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ######### Validation ###############
        val_data = datasets.ImageFolder(os.path.join(path, 'val'), transform=tf_image) 
        
        num_class = len(os.listdir(os.path.join(path, 'train')))

        print("Database report: \n")
        val_loader = thutils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
        print(val_data)

        return val_loader, num_class

def load_attacked_database_df(root_path, csv_path, batch_size, image_size=(128,128), percentage_attacked=0.1, test_size=None):
        tf_image = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if test_size is None:
            val = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="val")
            
            num_class = len(val.cl_name.values())
            
            val_loader = DataLoader(val, batch_size=batch_size, num_workers=4, shuffle=False)
        else:
            data = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path)
            
            train, test = train_test_split(list(range(len(data))), test_size=test_size, random_state=RANDOM_SEED)
            
            num_class = len(data.cl_name.values())
            
            index_num = int(np.floor(0.1*len(test)))
            val_index = test[len(test)-index_num:]
            
            sampler_val = Subset(data, val_index)
            
            val_loader = DataLoader(sampler_val, batch_size=batch_size, num_workers=4, shuffle=False)

        return val_loader, num_class

def show_images(dataset_loader, db_name):
    """function that show images from dataloader

    Args:
        dataset_loader (_type_): _description_
        db_name (_type_): _description_

    """
    batch = next(iter(dataset_loader))
    images, labels = batch
        
    plt.figure(figsize=(11, 11))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(images[:32], padding=2, normalize=True), (1, 2, 0)))
    plt.savefig("./attack-images/preview_train_{}.png".format(db_name))

class CustomDatasetFromCSV(Dataset):
    def __init__(self, path_root, tf_image, csv_name, task=None):
        self.data = pd.read_csv(csv_name)
        if task is not None:
            self.data.query("Task == @task", inplace=True)
        self.tf_image = tf_image
        self.root = path_root
        self.cl_name = {c: i for i, c in enumerate(np.unique(self.data["y"]))}
        self.BARVALUE = "/" if not os.name == "nt" else "\\"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #x_path = os.path.join(self.root, self.data.iloc[idx, 0].split(self.BARVALUE)[-2], self.data.iloc[idx, 0].split(self.BARVALUE)[-1])
        x_path = os.path.join(self.root, self.data.iloc[idx, 0])
        y = self.cl_name[self.data.iloc[idx, 1]]
        
        X = Image.open(x_path).convert("RGB")
 
        if self.tf_image:
            X = self.tf_image(X)
        
        return X, y