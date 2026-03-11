import torch
import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("./data3", split='train', download=True,
# transform=torchvision.transforms.ToTensor())

vgg16 = torchvision.models.vgg16(pretrained=False)

# 方式一
torch.save(vgg16, "vgg16_method1.pth")

# 方式二
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
