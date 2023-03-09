from pickletools import optimize
from pyexpat import model
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import sys
from efficientnet_pytorch import EfficientNet as EffNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channel = 3
path = "./BestModel.pth"
num_classes = 2
lr = 0.1
batch = 32
epochs = 5


class VClassification(torch.nn.Module):
    def __init__(self, input=100, output=1):
        super(VClassification, self).__init__()

        self.l1 = torch.nn.Linear(100, 20)
        self.l2 = torch.nn.Linear(20, 40)
        self.l3 = torch.nn.Linear(40, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inp):
        inp = self.l1(inp)
        inp = self.l2(inp)
        inp = self.l3(inp)
        inp = self.sigmoid(inp)
        return inp

def getModel(Name, OutFeatures):
    if Name=="effnet":
        model = EffNet.from_name("efficientnet-b0", in_channels=3)
        model._fc = nn.Linear(in_features=1280,out_features=OutFeatures)
    return model
model = getModel("effnet",3).to(device)
model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))["Model"])
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
for epoch in range(1):
  for _, label, feature_set in dataloader:

    label = label.unsqueeze(dim=1).double()
    preds = model(feature_set)
    preds = torch.where(preds>=0.5, 1.0, 0.0).double()
    preds = torch.tensor(preds, requires_grad=True)

    # print(label)
    # print(preds)

    loss = loss_function(preds, label)
    print(loss.item())
    loss.backward()
    optimizer.step()



