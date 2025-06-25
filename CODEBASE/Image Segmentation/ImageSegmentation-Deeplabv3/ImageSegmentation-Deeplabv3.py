import torch as t, os
from torchvision import models
from scikitlearn.model_selection import train_test_split

model = models.segmentation.deeplabv3_resnet50(pretrained=True)
images = ''

