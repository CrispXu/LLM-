import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("/mnt/dataset", train= False, transform= torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset= test_data,batch_size= 64, shuffle= False, num_workers= 0, drop_last= False)

#测试数据集中第一张图片
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2) :
    step = 0
    for data in test_loader :
        imgs, targets = data
        writer.add_images("epoch".format(epoch),imgs, step)
        #print(img.shape)
        #print(targets)
        step += 1
writer.close()