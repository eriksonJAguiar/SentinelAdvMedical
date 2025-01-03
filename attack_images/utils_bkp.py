
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
from PIL import Image
import torchvision

RANDOM_SEED = 43

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_attacked_database(path, batch_size, folder_from, image_size=(128,128)): 
        tf_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ######### Validation ###############
        val_data = datasets.ImageFolder(os.path.join(path, folder_from), transform=tf_image) 
        
        num_class = len(os.listdir(os.path.join(path, folder_from)))

        print("Database report: \n")
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
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
            val = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Val")
            
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
    plt.savefig("./attack-images/preview_train_{}.png".format(db_name), bbox_inches='tight', pad_inches=0)

def show_images_from_array(images_array, db_name):
    """function that show images from dataloader

    Args:
        dataset_loader (_type_): _description_
        db_name (_type_): _description_

    """        
    plt.figure(figsize=(11, 11))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(torch.Tensor(images_array[:32]), padding=2, normalize=True), (1, 2, 0)))
    plt.savefig("./attack-images/preview_train_{}.png".format(db_name), bbox_inches='tight', pad_inches=0)
    
def show_random_adv_image(images_array, db_name, attack_name):
    
    image_idx = np.random.randint(0, len(images_array))
    
    plt.figure(figsize=(11, 11))
    plt.axis("off")
    plt.imshow(np.transpose(make_grid(torch.Tensor(images_array[image_idx]), normalize=True), (1, 2, 0)))
    plt.savefig("./attack_preview_{}_{}.png".format(db_name, attack_name), bbox_inches='tight', pad_inches=0)
    
def save_all_adv_image(path_to_save, images_array, labels, db_name , attack_name, cls=["akiec","bcc","bkl","df","mel","nv","vasc"]):
    
    if not os.path.exists(os.path.join(path_to_save, db_name, attack_name)):
        os.mkdir(os.path.join(path_to_save, db_name, attack_name))
    
    for c in cls:
        if not os.path.exists(os.path.join(path_to_save, db_name, attack_name, c)):
            os.mkdir(os.path.join(path_to_save, db_name, attack_name, c))
    
    plt.figure(figsize=(11, 11))
    plt.axis("off")
    for i in range(len(images_array)):
        plt.axis("off")
        plt.imshow(np.transpose(make_grid(torch.Tensor(images_array[i]), normalize=True), (1, 2, 0)))
        plt.savefig("{}/{}/{}/{}/{}_{}_{}.png".format(path_to_save, db_name, attack_name, cls[int(labels[i])], cls[int(labels[i])], attack_name, i), bbox_inches='tight', pad_inches=0)
    
def read_model_from_checkpoint(model_path, model_name, nb_class):

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {key[6:] : checkpoint['state_dict'][key] for key in checkpoint['state_dict']}
    model = __get_model_structure(model_name, nb_class)
    model.load_state_dict(state_dict)
    
    return model

def __get_model_structure(model_name, nb_class):
    model = None
    #"resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
    if model_name == "resnet50":
        model = torchvision.models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class)
        )

    elif model_name == "vgg16":
        model = torchvision.models.vgg.vgg16()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )
    
    elif model_name == "vgg19":
        model = torchvision.models.vgg.vgg19()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )
    
    elif model_name == "inceptionv3":
        model = torchvision.models.inception_v3()
        model.aux_logits = False
        
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )       
    
    elif model_name == "efficientnet":
        model = torchvision.models.efficientnet_b0()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )
    
    elif model_name == "densenet":
        model = torchvision.models.densenet121()
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )
    
    return model

def numpy_to_dataloader(images, labels, batch_size):
    
    dataset  = CustomDataset(images, labels)
    
    val_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    return val_loader

def dataloader_to_numpy(dataloader):
    
    images, labels = zip(*[dataloader.dataset[i] for i in range(len(dataloader.dataset))])
    images = torch.stack(images).numpy() 
    labels = np.array(labels)
    
    return images, labels

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
    
class CustomDataset(Dataset):
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        X = self.images[idx]
        y = self.labels[idx]
        
        # if self.transform:
        #     x = Image.fromarray(self.data[idx].transpose(1,2,0))
        #     x = self.transform(x)
        
        return X, y