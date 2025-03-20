import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10("/mnt/dataset", train = False, transform=torchvision.transforms.ToTensor(),download = True)

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608,10)

    def forward(self, input):
        output = self.linear1(input)
        return output
    
tudui = Tudui()

step = 0
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)
