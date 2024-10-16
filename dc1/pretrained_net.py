import torch
import torch.nn as nn
from torchvision import models 
from torchsummary import summary
from torchvision.models.googlenet import BasicConv2d

# Pretrained GoogLeNet 
google_model = models.googlenet(pretrained=True)
google_model.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
google_model.fc = nn.Linear(1024, 6)

