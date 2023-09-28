from torch import save, utils as thutils
from torchvision import transforms, datasets, utils
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
import torcheval.metrics.functional as tm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import medmnist
import PIL
from medmnist import INFO, Evaluator
from sklearn.model_selection import train_test_split
from collections import Counter

class UtilsTroch:
    
    @staticmethod
    def load_database(path, batch_size, image_size=(128,128), is_agumentation=False):
        if is_agumentation:
            tf_image = transforms.Compose([#transforms.ToPILImage(),
                                       transforms.Resize(image_size),
                                       transforms.RandomHorizontalFlip(0.3),
                                       transforms.RandomVerticalFlip(0.3),
                                       transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        else:
            tf_image = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        ######### Train ###############
        train_data = datasets.ImageFolder(os.path.join(path, 'train'), transform=tf_image)

        ######### test ###############
        test_data = datasets.ImageFolder(os.path.join(path, 'test'), transform=tf_image)

        # ######### Validation ###############
        # val_data = datasets.ImageFolder(os.path.join(path, 'val'), transform=tf_image) 
        
        num_class = len(os.listdir(os.path.join(path, 'train')))

        print("Database report: \n")
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8)
        print(train_data)

        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=8)
        print(test_data)

        # val_loader = thutils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
        # print(val_data)

        return train_loader, test_loader, num_class

    @staticmethod
    def load_images_path(img_dir, image_size = (128, 128)):
        tf_image = transforms.Compose([transforms.ToTensor(),  
                                       transforms.Resize(image_size), 
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])

        database = ImageFolder(root=img_dir, transform=tf_image)
        
        return database

    @staticmethod
    def load_database_kf(path_image, batch_size, image_size=(128,128),  n_folds=5, csv_path=None):
        tf_image = transforms.Compose([#transforms.ToPILImage(),
                                       transforms.Resize(image_size),
                                       transforms.RandomHorizontalFlip(0.3),
                                       transforms.RandomVerticalFlip(0.3),
                                       transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
                                       transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        
        kf = KFold(n_splits=n_folds, shuffle=True)
        #kf = StratifiedKFold(n_splits=n_folds)
        train_loader, test_loader = {}, {}
        database = None
        num_class = 0
        if csv_path is None:
            database = datasets.ImageFolder(path_image, transform=tf_image)
            num_class = len(os.listdir(path_image))
        else:
            database = CustomDatasetFromCSV(root=path_image, tf_image=tf_image, csv_name=csv_path)
            num_class = len(database.cl_name.values())
        
        for i, (train_index, test_index) in enumerate(kf.split(database)):
                
                train_sampler = SubsetRandomSampler(train_index)
                test_sampler = SubsetRandomSampler(test_index)
                
                train_loader[i] = DataLoader(database, batch_size=batch_size, sampler=train_sampler, num_workers=8)
                test_loader[i] = DataLoader(database, batch_size=batch_size, sampler=test_sampler, num_workers=8)
        
        # print("Database report: \n")
        # train_loader = thutils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        # print(train_data)

        # test_loader = thutils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        # print(test_data)

        # val_loader = thutils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
        # print(val_data)

        return train_loader, test_loader, num_class
    
    
    @staticmethod
    def load_database_df(root_path, csv_path, batch_size, image_size=(128,128), is_agumentation=False, test_size=None):
        if is_agumentation:
            tf_image = transforms.Compose([#transforms.ToPILImage(),
                                       transforms.Resize(image_size),
                                       transforms.RandomHorizontalFlip(0.3),
                                       transforms.RandomVerticalFlip(0.3),
                                       transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        else:
            tf_image = transforms.Compose([
                transforms.Resize(image_size),
                transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        def get_sampler(train, test):
            
            train_values = dict(Counter(sorted(list(map(lambda x: x[1], train)))))
            test_values = dict(Counter(sorted(list(map(lambda x: x[1], test)))))
                
            class_weights_train = 1.0/np.array([*train_values.values()])
            class_weights_test = 1.0/np.array([*test_values.values()])
                
            sampler_train = WeightedRandomSampler(weights=class_weights_train, replacement=False, num_samples=len(train_values))
            sampler_test = WeightedRandomSampler(weights=class_weights_test, replacement=False, num_samples=len(test_values))
            
            return sampler_train, sampler_test
        
        if test_size is None:
            train = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Train")
            test = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Test")
            num_class = len(train.cl_name.values())
            
            #train_sampler, test_sampler = get_sampler(train, test)
    
            #train_loader = DataLoader(train, batch_size=batch_size, num_workers=8, sampler=train_sampler)
            #test_loader = DataLoader(test, batch_size=batch_size, num_workers=8, )
            
            train_loader = DataLoader(train, batch_size=batch_size, num_workers=8, shuffle=True)
            test_loader = DataLoader(test, batch_size=batch_size, num_workers=8, shuffle=False)
        else:
            data = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path)
            
            train, test = train_test_split(data, test_size=test_size, shuffle=True)
            # dataset_size = data.__len__()
            # train_count = (dataset_size*(1-test_size))
            # test_count = dataset_size - train_count
            # train, test = thutils.data.random_split(data, [train_count, test_count])
            # train_sampler, test_sampler = get_sampler(train, test)
            # exit(0)
            
            num_class = len(data.cl_name.values())
            train_loader = DataLoader(train, batch_size=batch_size, num_workers=8, sampler=train_sampler)
            test_loader = DataLoader(test, batch_size=batch_size, num_workers=8, sampler=test_sampler)
        
        
        # print("Database report: \n")
        # train_loader = thutils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        # print(train_data)

        # test_loader = thutils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        # print(test_data)

        # val_loader = thutils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
        # print(val_data)

        return train_loader, test_loader, num_class
    
    @staticmethod
    def load_medmnist(database_name, image_size, batch_size, as_rgb, test_size=0.2, balanced=True, augmentation=False):
        
        if augmentation:
            f_image = transforms.Compose([#transforms.ToPILImage(),
                                       transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
                                       transforms.Grayscale(num_output_channels=3),
                                       transforms.RandomHorizontalFlip(0.3),
                                       transforms.RandomVerticalFlip(0.3),
                                       transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                       #transforms.Normalize(mean=[0.5], std=[0.5])
                                    ])
        else:
            f_image = transforms.Compose([#transforms.ToPILImage(),
                                        transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
                                        #transforms.Grayscale(num_output_channels=3),
                                        #transforms.RandomHorizontalFlip(0.3),
                                        #transforms.RandomVerticalFlip(0.3),
                                        #transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        #transforms.Normalize(mean=[0.5], std=[0.5])
                                        ])
        
        
        data_flag = database_name.lower()

        info = INFO[data_flag]
        task = info['task']
        print("Task: {}".format(task))
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        
        DataClass = getattr(medmnist, info["python_class"])
        # train = None
        # if database_name.lower() == "breastmnist":
        #     train = medmnist.BreastMNIST(root="../dataset/medmnist", split="train", transform=f_image, download=True)
        # elif database_name.lower() == "chestmnist":
        #     train = medmnist.ChestMNIST(root="../dataset/medmnist", split="train", transform=f_image, download=True)
        # #elif database_name.lower() == ""
        # else:
        #     print("dataset not found")
        #     exit(0)

        #train = medmnist.ChestMNIST(split="train", transform=f_image, download=True)
        #test = medmnist.ChestMNIST(split="test", transform=f_image, download=True)
        train_data = DataClass(root="../dataset/medmnist", split="train", transform=f_image, download=True, as_rgb=as_rgb)
        test_data =  DataClass(root="../dataset/medmnist", split="test", transform=f_image, download=True, as_rgb=as_rgb)
        
        if not balanced:
            train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=False)
        
        else:
            counter_train = dict(Counter(train_data.labels.flatten()))
            print(counter_train)
            
            targets_train = train_data.labels.flatten()
            class_unique = list(counter_train.keys())
            class_df_train = pd.DataFrame({"index": range(0, len(targets_train)), "label": targets_train})
            key_min = min(counter_train, key=counter_train.get)
            min_value = counter_train[key_min]
            
            index_train = np.array([np.random.choice(list(class_df_train[class_df_train["label"] == c]["index"]), min_value, replace=False) for c in class_unique]).flatten()
            
            
            counter_test = dict(Counter(test_data.labels.flatten()))
            print(counter_test)
            
            targets_test = test_data.labels.flatten()
            class_df_test = pd.DataFrame({"index": range(0, len(targets_test)), "label": targets_test})
            key_min = min(counter_test, key=counter_test.get)
            min_value = counter_test[key_min]
            
            index_test = np.array([np.random.choice(list(class_df_test[class_df_test["label"] == c]["index"]), min_value, replace=False) for c in class_unique]).flatten()
            
            train_sampler = SubsetRandomSampler(index_train)
            test_sampler = SubsetRandomSampler(index_test)
        
            train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8, sampler=train_sampler)
            test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8, sampler=test_sampler)
        
        return train_loader, test_loader, n_classes, task
    
    @staticmethod
    def load_medmnist_split(database_name, image_size, batch_size, as_rgb, test_size=0.2):
        
        data_flag = database_name.lower()
        info = INFO[data_flag]
        task = info['task']
        print("Task: {}".format(task))
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        
        train_data = DatasetMedmnistCustom(dataset_name=data_flag, image_size=image_size, as_rgb=as_rgb)
        test_data = DatasetMedmnistCustom(dataset_name=data_flag, image_size=image_size, as_rgb=as_rgb, is_test=True)
        
        train_index = list(range(train_data.len_dataset))
        test_index = list(range(test_data.len_dataset))
        
        train_size = int(train_data.len_dataset*(1-test_size))
        print(train_size)
        test_size = int(train_size*test_size)
        print(test_data.len_dataset/train_data.len_dataset)
        print(test_size)
        #train_index = np.random.choice(train_index, train_size)
        #test_index = np.random.choice(train_index, train_size)
    
    @staticmethod
    def show_images(dataset, batch_size):
        loader = thutils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        batch = next(iter(loader))
        images, labels = batch
        grid = utils.make_grid(images, nrow=8)
        plt.figure(figsize=(11,11))
        plt.axis('off')
        plt.imshow(np.transpose(np.clip(grid, 0, 1), (1,2,0)))
        plt.savefig("preview_images.png")
        print(labels)

class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            save(model.state_dict(), model_path)
        self.val_score = epoch_score
        
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        #self.y = self.y.float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        return x, y

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
    
class DatasetFromFolder(Dataset):
    def __init__(self, img_dir, image_size=(128,128)):
        tf_image = transforms.Compose([ transforms.Resize(image_size),
                                        transforms.ToTensor(),  
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        self.dataset = ImageFolder(root=img_dir, transform=tf_image)
        self.X = []
        self.y = []
        
        self.X, self.y = zip(*[(x, y) for x, y in self.dataset])
        
        # for img, y in self.dataset:
        #     self.X.append(img)
        #     self.y.append(y)
        
        self.X = torch.stack(list(self.X))
        self.y = torch.tensor(list(self.y))
        
        #self.y = self.y.unsqueeze(-1)
        
        # print('Shape of x:', self.X.shape)
        # print('Shape of y:', self.y.shape)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        return x, y


class DatasetMedmnistCustom(Dataset):
    
    def __init__(self, dataset_name, image_size=(128,128), as_rgb=True, is_test=False):
        tf_image = transforms.Compose([ transforms.Resize(image_size),
                                        transforms.ToTensor(),  
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        
        data_flag = dataset_name.lower()
        info = INFO[data_flag]
        self.n_classes = len(info['label'])
        DataClass = getattr(medmnist, info["python_class"])
        
        
        if is_test:
            self.__data = DataClass(root="../dataset/medmnist", split="test", transform=tf_image, download=True, as_rgb=as_rgb)
        else:
            self.__data = DataClass(root="../dataset/medmnist", split="train", transform=tf_image, download=True, as_rgb=as_rgb)
            
        
        self.len_dataset = len(self.__data)
        
        #self.__test_data =  DataClass(root="../dataset/medmnist", split="test", transform=tf_image, download=True, as_rgb=as_rgb)
        
        #val = {i: methodcaller(i)(df['close']) for i in stat}
        
        
        
        # self.X = []
        # self.y = []
        
        # #self.X, self.y = zip(*[(x, y) for x, y in self.dataset])
        
        # # for img, y in self.dataset:
        # #     self.X.append(img)
        # #     self.y.append(y)
        
        # self.X = torch.stack(list(self.X))
        # self.y = torch.tensor(list(self.y))
        
        #self.y = self.y.unsqueeze(-1)
        
        # print('Shape of x:', self.X.shape)
        # print('Shape of y:', self.y.shape)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.__data[idx][0]
        y = self.__data[idx][1]
        
        return x, y


class MetricsClassification():

    def __init__(self):
        pass
    
    def __calculate_confuse_matrix(self, y_pred, y_true, num_class):
        #mt = confusion_matrix(y_true, y_pred, labels=labels).ravel()
        
        if  num_class > 2:
            fp = torch.logical_and(y_true != y_pred, y_pred != -1).sum()
            fn = torch.logical_and(y_true != y_pred, y_pred == -1).sum()
            tp = torch.logical_and(y_true == y_pred, y_true != -1).sum()
            tn = torch.logical_and(y_true == y_pred, y_true == -1).sum()
            mt = (tn, fp, fn, tp)
        else: 
           mt = tm.binary_confusion_matrix(y_pred, y_true).ravel()
        
        tn, fp, fn, tp = mt
        
        return tn, fp, fn, tp
    
    def accuracy_metric(self, y_true, y_pred, labels):
        tn, fp, fn, tp = self.__calculate_confuse_matrix(y_true, y_pred, labels)
        
        if (tp+tn+fp+fn) == 0:
            return 0.0
        
        return (tp+tn)/(tp+tn+fp+fn)
    
    def precision_metric(self, y_true, y_pred, labels):
        tn, fp, fn, tp = self.__calculate_confuse_matrix(y_true, y_pred, labels)
        
        if (tp+fp) == 0:
            return 0.0
        
        return tp/(tp+fp)
    
    def recall_metric(self, y_true, y_pred, labels):
        tn, fp, fn, tp = self.__calculate_confuse_matrix(y_true, y_pred, labels)
        
        if (tp+fn) == 0:
            return 0.0
        
        return tp/(tp+fn)
    
    def specificity_metric(self, y_pred, y_true, num_class):
        tn, fp, fn, tp = self.__calculate_confuse_matrix(y_pred, y_true, num_class)
        
        return tn/(tn+fp)
    
    def mse_metric(self, y_pred, y_true, num_class):
        n = len(y_pred)
        
        mse = (1/n)*torch.sum((y_pred - y_true)**2)
        
        return mse
    
    def f1_metric(self, y_true, y_pred, labels):
        re = self.recall_metric(y_true, y_pred, labels)
        pr = self.precision_metric(y_true, y_pred, labels)
    
        if (pr+re) == 0:
            return 0.0
        
        return 2*((pr*re)/(pr+re))
    
    
    def rocauc_metric(self, y_true, y_pred, labels):
        sns = self.recall_metric(y_true, y_pred, labels)
        spc = self.specificity_metric(y_true, y_pred, labels)
        
        return (np.sqrt(sns**2 + spc**2)/np.sqrt(2)) 