import torch
import torchvision

# 方式一
model = torch.load("vgg16_method1.pth", weights_only=False)

# 方式二
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
