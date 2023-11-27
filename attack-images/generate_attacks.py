import load_dataset as ld
import torch
import torchvision

def generate_attack(model_path, model_name, validation_attack_loader, input_shape, nb_class, attack_name, eps):
    
    #1st: read a pytorch model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {key[6:] : checkpoint['state_dict'][key] for key in checkpoint['state_dict']}
    model = __get_model_structure(model_name)
    model.load_state_dict(state_dict)
    
    #2nd define the loss and 
    
    
    #3rd 
    
    #4th
    
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