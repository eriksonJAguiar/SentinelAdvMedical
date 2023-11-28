import load_dataset as ld
import torch
import torchvision
import numpy as np

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, DeepFool, CarliniL2Method, UniversalPerturbation

def generate_attack(model_path, model_name, data_loader, input_shape, lr, nb_class, attack_name, eps):
    
    #1st: read a pytorch model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {key[6:] : checkpoint['state_dict'][key] for key in checkpoint['state_dict']}
    model = __get_model_structure(model_name)
    model.load_state_dict(state_dict, strict=False)
    
    #2nd define the loss and optimizer
    loss = torch.nn.CrossEntropyLoss() if nb_class > 2 else torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    #3rd create ART classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=opt,
        input_shape=input_shape,
        nb_classes=nb_class
    )
    
    #4th generate attack
    x_adv_examples, true_labels = __get_adv_attack(attack_name=attack_name, data_loader=data_loader, classifier=classifier, eps=eps)
    
    return x_adv_examples, true_labels
    
def __get_model_structure(model_name):
    model = None
    #"resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
    if model_name == "resnet50":
        model = torchvision.models.resnet50()

    elif model_name == "vgg16":
        model = torchvision.models.vgg.vgg16()
    
    elif model_name == "vgg19":
        model = torchvision.models.vgg.vgg19()
    
    elif model_name == "inceptionv3":
        model = torchvision.models.inception_v3()
    
    elif model_name == "efficientnet":
        model = torchvision.models.efficientnet_b0()
    
    elif model_name == "densenet":
        model = torchvision.models.densenet121()
    
    return model


def __get_adv_attack(attack_name, data_loader, classifier, eps):
    
    attack = None
    #FGSM, DeepFool, C&W, UAP
    if attack_name == "FGSM":
        attack = FastGradientMethod(estimator=classifier, eps=eps)
    elif attack_name == "DeepFool":
        attack = DeepFool(classifier=classifier, epsilon=eps)
    elif attack_name == "CW":
        attack = CarliniL2Method(classifier=classifier)
    elif attack_name == "UAP":
        attack_name = UniversalPerturbation(classifier=classifier, eps=eps)
        
    images, labels = zip(*[data_loader.dataset[i] for i in range(len(data_loader.dataset))])
    images = torch.stack(images).numpy()
    true_labels = np.array(labels)
    
    adv_attack = attack.generate(x=images, y=true_labels)
    
    return adv_attack, true_labels