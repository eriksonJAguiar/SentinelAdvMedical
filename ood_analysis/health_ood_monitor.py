from pytorch_ood.utils import extract_features
from pytorch_ood.api import Detector
from sklearn.neighbors import KernelDensity
import torch
import sys

sys.path.insert(0, "../attack_images")
from attack_images.evaluate import get_last_layer_features

# class HealthOODMonitor():
    
#     def __init__(self) -> None:
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     def kernel_density(self, dataloader, model):
        
#         z, y = extract_features(dataloader, model, self.device)
        
#         kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(z)
        
#         return kde


class KDE():
    
    def __init__(self, model) -> None:
        super(KDE, self).__init__()
        self.model = model
    
    def __call__(self, *args, **kwargs):
        """
        Forwards to predict
        """
        return self.predict(*args, **kwargs)
    
    
    def fit(self, dataloader, device):
        
        z, y = extract_features(dataloader, self.model, device)
        
        print(z[0])
        
        print(z.shape)
        
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(z.cpu().numpy())
        
        return self
        
    
    def predict(self, x):
        
       z = torch.Tensor()
       
       get_last_layer_features(self.model, x)
       
       scores = self.kde.score_samples(x.cpu().numpy())
       
       return torch.Tensor(scores)