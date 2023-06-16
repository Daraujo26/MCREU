from einops.layers.torch import Rearrange, Reduce
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision.utils import save_image
import torchvision
import matplotlib.pyplot as plt



device = "cuda"

dataset_train = torchvision.datasets.CIFAR10(".", train=True, download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.4914, 0.4822, 0.4465)),
]))
dataset_test = torchvision.datasets.CIFAR10(".", train=False, download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.4914, 0.4822, 0.4465)),
]))

dataloader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=4096)

for (images, labels) in dataloader_train:
    # y = (y - y.max()) / (y.max() - y.min())
    # if y is a tensor then the code above will normalize y such that
    # the new y.min() is zero and the new y.max() is one.

    images = (images - images.max()) / (images.max() - images.min())

    save_image(images, "image.png")
    exit()

import torch as T
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
  img = img / 2 + 0.5   # unnormalize
  npimg = img.numpy()   # convert from tensor
  plt.imshow(np.transpose(npimg, (1, 2, 0))) 
  plt.show()

def main():
  transform = transforms.Compose( [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
      (0.5, 0.5, 0.5))])

  trainset = tv.datasets.CIFAR10(root='.\\data', train=True,
    download=False, transform=transform)
  trainloader = T.utils.data.DataLoader(trainset,
    batch_size=100, shuffle=False, num_workers=1)

  # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
  #   'frog', 'horse', 'ship', 'truck')

  # get first 100 training images
  dataiter = iter(trainloader)
  imgs, lbls = dataiter.next()

  for i in range(100):  # show just the frogs
    if lbls[i] == 6:  # 6 = frog
      imshow(tv.utils.make_grid(imgs[i]))

if __name__ == "__main__":
  main()