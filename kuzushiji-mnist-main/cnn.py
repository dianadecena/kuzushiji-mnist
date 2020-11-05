import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, modelo):
        super(ConvNet, self).__init__()
        self.modelo = modelo
        if self.modelo == 1:
            self.layer1 = nn.Sequential( # input = (100*1*28*28)
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0), # output = (100*6*24*24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)) # output = (100*6*12*12)
            self.layer2 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=4, stride=1, padding=0), # output = (100*12*9*9)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)) # output = (100*12*3*3)
            self.fc1 = nn.Linear(108, 10)

    def forward(self, x):
        if self.modelo == 1:
            l1 = self.layer1(x)
            l2 = self.layer2(l1)
            l2 = l2.view(l2.size(0), -1)
            l3 = self.fc1(l2)
            output = F.softmax(l3, dim=1)
            return output