import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch

#定义训练的设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#准备数据集
train_data = torchvision.datasets.CIFAR10("/mnt/dataset",
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("/mnt/dataset",
                                          train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
#length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

#利用Dataloader来加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x 

#创建网络模型
tudui = Tudui()
tudui.to(device)

#损失函数
loss_function = nn.CrossEntropyLoss()
loss_function.to(device)

#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)

#设置训练网络参数
total_train_step = 0 #记录训练的次数
total_test_step = 0 #记录测试的次数
epoch = 10 #训练的轮次

#添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("第{}轮训练开始".format(i+1))

    #训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_function(outputs, targets)
    
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss += loss.item()
            accuracy = sum(outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(tudui,"tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
