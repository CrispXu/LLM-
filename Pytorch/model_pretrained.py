import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)

#print(vgg16_false)

train_data = torchvision.datasets.CIFAR10("/mnt/dataset", train = False, transform=torchvision.transforms.ToTensor(),download = True)

#增加层
vgg16_false.classifier.add_module('add_linear',nn.Linear(1000,10))

#修改层
vgg16_false.classifier[6] = nn.Linear(4096, 10)

print(vgg16_false)
