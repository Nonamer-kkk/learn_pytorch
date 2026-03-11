import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("./data3", split='train', download=True,
# transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_false)
print(vgg16_true)
vgg16_true.classifier.add_module('add_Linear', nn.Linear)
vgg16_false.classifier[6] = nn.Linear(4096, 10)