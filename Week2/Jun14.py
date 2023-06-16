# Print the first 10 images from CIFAR10 training dataset and first 10 from testing dataset where each image 
# is separetley displayed in a window.
# Each figure needs to be titled with the corresponding label in words (Not the number 0-9)
import torchvision
import matplotlib.pyplot as plt
import numpy as np

list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=True)
testset = torchvision.datasets.CIFAR10(root='.', train=False, download=True)

for i in range(10):
    x, y = trainset[i]
    x = np.asarray(x)
    plt.imshow(x)
    plt.title(list[y])
    plt.show()

    y, w = testset[i]
    y = np.asarray(y)
    plt.imshow(y)
    plt.title(list[w])
    plt.show()

