from torchvision import transforms
from PIL import Image 
from torch.utils.tensorboard import SummaryWriter
img_path = "/mnt/learn_torch/Dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer = SummaryWriter("logs")
writer.add_image("Tensor_img", tensor_img)
writer.close()