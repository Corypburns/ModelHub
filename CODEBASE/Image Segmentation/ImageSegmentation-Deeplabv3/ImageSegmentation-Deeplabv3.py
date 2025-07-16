import torch as t, os
from torchvision import models
from sklearn.model_selection import train_test_split as tts

model = models.segmentation.deeplabv3_resnet50(pretrained=True)
images = ''