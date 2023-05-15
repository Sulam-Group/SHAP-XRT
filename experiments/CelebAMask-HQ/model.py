import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 1)

        self.resnet = resnet
        self.act = torch.sigmoid

    def forward(self, x):
        x = self.resnet(x)
        x = self.act(x)
        return x
