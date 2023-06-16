import torch
import torchvision as tv
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
  img = img / 2 + 0.5
  npimg = img.numpy() 
  plt.imshow(np.transpose(npimg, (1, 2, 0))) 
  plt.show()

def main():
  transform = transforms.Compose( [transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
      (0.4914, 0.4822, 0.4465))])

  trainset = tv.datasets.CIFAR10(root='./data', train=True,
    download=True, transform=transform)
  trainloader = DataLoader(trainset,
    batch_size=100, shuffle=False, num_workers=1)

  dataiter = iter(trainloader)
  imgs, lbls = next(dataiter)

  for i in range(100):
    if lbls[i] == 9:  
      imshow(tv.utils.make_grid(imgs[i]))

if __name__ == "__main__":
  main()