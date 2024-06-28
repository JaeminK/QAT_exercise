import torch
import torch.nn as nn

from utils import QuantConv2d, QuantLinear

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(32 * 32 * 32, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
class QCNN(nn.Module):
    def __init__(self, CNN):
        super(QCNN, self).__init__()
        self.q_conv1 = QuantConv2d(CNN.conv1, CNN.bn1, activation=CNN.relu1)
        self.q_conv2 = QuantConv2d(CNN.conv2, CNN.bn2, activation=CNN.relu2)
        self.q_fc = QuantLinear(CNN.fc, activation=None)
        
    def forward(self, x):
        x = self.q_conv1(x)
        x = self.q_conv2(x)
        x = x.view(x.size(0), -1)
        x = self.q_fc(x)
        return x