import torch


"""
    Summary of Inception-Resnet-v1
    Input -> (299,299,3)
    Steam -> output: (35x35x384)
    4 x Inception-A -> output: (35x35x384)
    Reduction-A -> output: (17x17x1024)
    7 x Inception-B -> output: 17x17x1024
    Reduction-B -> output: 8x8x1536
    3 x Inception-C -> output: 8x8x1536
    Average Pooling -> output: 1536
    Dropout (keep=0.8) -> output: 1536
    Linear -> output: number of classes or Softmax -> output: number of classes
"""


class Steam(torch.nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.part1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.part2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=80, kernel_size=1),
            torch.nn.Conv2d(in_channels=80, out_channels=192, kernel_size=3),
            torch.nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3),
        )

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        
        return x
    

class Inception_ResNet_A(torch.nn.Module):
    
    def __init__(self, in_channel, scale):
        super().__init__()
        self.scale = scale
        self.part1 = torch.nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=1, stride=1, padding=0)
        
        self.part2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
        )
        
        self.part3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
        )
        self.conv_final = torch.nn.Conv2d(in_channels=160, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=True) 

    def forward(self, x):
        x0 = self.part1(x)
        x1 = self.part2(x)
        x2 = self.part3(x)
        
        x_final = torch.cat((x0,x1,x2), dim=1)
        x_final = self.conv_final(x_final)
        
        return self.relu(x + x_final*self.scale)    
    
    
class Inception_ResNet_B(torch.nn.Module):
          
    def __init__(self, in_channel, scale):
        super().__init__()
        self.scale = scale
        
        self.part1 = torch.nn.Conv2d(in_channels=in_channel, out_channels=192, kernel_size=1)
        
        self.part3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=128, kernel_size=1),
            torch.nn.Conv2d(in_channels=128, out_channels=160, kernel_size=(1,7)),
            torch.nn.Conv2d(in_channels=160, out_channels=192, kernel_size=(7,1)),
        )
        self.conv_final = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x0 = self.part1(x)
        x1 = self.part2(x)
        
        x_final = torch.cat((x0,x1), dim=1)
        x_final = self.conv_final(x_final)
        
        return self.relu(x + x_final*self.scale) 
    
    

class Inception_ResNet_C(torch.nn.Module):
          
    def __init__(self, in_channel, scale):
        super().__init__()
        self.scale = scale
        
        self.part1 = torch.nn.Conv2d(in_channels=in_channel, out_channels=192, kernel_size=1)
        
        self.part3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=192, kernel_size=1),
            torch.nn.Conv2d(in_channels=192, out_channels=224, kernel_size=(1,3)),
            torch.nn.Conv2d(in_channels=224, out_channels=256, kernel_size=(3,1)),
        )
        self.conv_final = torch.nn.Conv2d(in_channels=448, out_channels=2048, kernel_size=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x0 = self.part1(x)
        x1 = self.part2(x)
        
        x_final = torch.cat((x0,x1), dim=1)
        x_final = self.conv_final(x_final)
        
        return self.relu(x + x_final*self.scale)  
    

class ReductionB(torch.nn.Module):
    
    def __init__(self, in_channel):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.part1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, kernel_size=1, out_channels=256, stride=3),
            torch.nn.Conv2d(in_channels=256, kernel_size=3, out_channels=384, stride=2),
        )  
        
        self.part2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=256, kernel_size=1),
            torch.nn.Conv2d(in_channels=256, out_channels=288, kernel_size=3, stride=2),
        )
        
        self.part3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=256, kernel_size=1),
            torch.nn.Conv2d(in_channels=256, out_channels=288, kernel_size=3),
            torch.nn.Conv2d(in_channels=288, out_channels=320, kernel_size=3, stride=2),
        )
        
    
    def forward(self, x):
        x0 = self.pool(x)
        x1 = self.part1(x)
        x2 = self.part2(x)
        x3 = self.part3(x)
        
        return torch.cat((x0, x1, x2, x3), dim=1)
    


class ReductionA(torch.nn.Module):
    
    def __init__(self, in_channel, k, l, m, n):
        super().__init__()
        self.part1 = torch.nn.Conv2d(in_channels=in_channel, out_channels=n, kernel_size=3, stride=2)
        
        self.part1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=k, kernel_size=1, stride=1),
            torch.nn.Conv2d(in_channels=k, out_channels=l, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=l, out_channels=m, kernel_size=3, stride=2),
        )  
        
        self.part2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=256, kernel_size=1),
            torch.nn.Conv2d(in_channels=256, out_channels=288, kernel_size=3, stride=2),
        )
        
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
    
    def forward(self, x):
        x0 = self.part1(x)
        x1 = self.part2(x)
        x2 = self.pool(x)
        
        return torch.cat((x0, x1, x2), dim=1)
    


class Resnet_Inception_v1(torch.nn.Module):
    
    def __init__(self, in_channel=3, scale=1.0, classes=1000, k=192, l=192, m=256, n=384):
        super().__init__()
        blocks = []
        
        blocks.append(Steam(in_channel=in_channel))
        
        for i in range(5):
            blocks.append(Inception_ResNet_A(256, 0.17))
        
        blocks.append(ReductionA(256, k, l, m, n))
        
        for i in range(10):
            blocks.append(Inception_ResNet_B(1088, 0.10))
        
        blocks.append(ReductionB(1088))
        
        for i in range(5):
            blocks.append(Inception_ResNet_C(2080, 0.20))
        
        self.features = torch.nn.Sequential(*blocks)
        self.pool = torch.nn.AvgPool2d(kernel_size=(1,1))
        self.conv = torch.nn.Conv2d(2080, 1536, 1, stride=1)
        self.dropout = torch.nn.Dropout(0.8)
        self.linear = torch.nn.Linear(in_features=1536, out_features=classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        return x
