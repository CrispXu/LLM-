from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("/mnt/learn_torch/images/IMG_20250119_155244.jpg")

#ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

#Normalize
trans_norm = transforms.Normalize([1,2,5],[2,4,8])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm,1)

#Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
print(img_resize)
writer.add_image("Resize", img_resize,0)

#Compose - Resize
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2, 1)

#RandomCrop
trans_radom = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_radom, trans_totensor])
for i in range(10) :
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)
writer.close()