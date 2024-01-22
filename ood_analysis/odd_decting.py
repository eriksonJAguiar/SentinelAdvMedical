import sys
sys.path.append("../attack_images")
sys.path.append("../utils")
sys.path.append("./health_ood_monitor")


from utils import utils
from pytorch_ood.detector import MaxSoftmax, ODIN, MaxLogit, Mahalanobis, EnergyBased, KNN, ViM, Entropy, MCD, OpenMax
from pytorch_ood.utils import OODMetrics
from health_ood_monitor import KDE
import numpy as np
import os


def odd_detector(weights_path, model_name, dataset_name, in_images, out_images, in_random_images, batch_size, ood_name, eps, nb_class):
    
    
    in_out_images = np.concatenate((in_images, out_images), axis=0)
    in_out_labels = np.concatenate((np.repeat(0, len(in_images)), np.repeat(-1, len(out_images))), axis=0)
    
    #4th convert images and labels to dataloader
    dataloader_atttacked = utils.numpy_to_dataloader(images=in_out_images, labels=in_out_labels, batch_size=batch_size)
    
    model_path = os.path.join(weights_path, "{}-{}-exp1.ckpt".format(model_name, dataset_name))
    model = utils.read_model_from_checkpoint(model_path=model_path, model_name=model_name, nb_class=nb_class)
    model.eval().cuda()
    
    #5th create the detector
    dectetor = __get_ood_strategy(ood_name=ood_name, dataloader=in_random_images, model=model, eps=eps, t=3.0)
    
    #6th calculate metrics for detector
    metrics = OODMetrics()
    
    for x, y in dataloader_atttacked:
        #print(dist(x.cuda()))
        metrics.update(dectetor(x.cuda()), y)
        print(dectetor(x.cuda()))
    #features = dectetor.fit(dataloader_atttacked, device="cuda")
    # print(dectetor.predict(features))
    # metrics.update(dectetor.predict(features))
    
    odd_metrics = dict(metrics.compute())
    
    return odd_metrics


def __get_ood_strategy(ood_name, dataloader, model, t=1.0, eps=0.01):
    
    ood_strategies = {
        "MaxSoftmax": MaxSoftmax(model=model, t=t),
        "ODIN": ODIN(model=model, eps=eps, temperature=t),
        "MaxLogit": MaxLogit(model=model),
        "Mahalanobis": Mahalanobis(model=model, eps=eps),
        "Entropy": Entropy(model=model),
        "KNN": KNN(model=model),
        "MCD": MCD(model=model),
        "KD": KDE(model=model)
    }
    
    ood_method = ood_strategies[ood_name]
    if ood_name == "KNN" or ood_name == "Mahalanobis" or ood_name == "KD":
        ood_method = ood_strategies[ood_name]
        ood_method = ood_method.fit(dataloader, device="cuda")
    
    return ood_method