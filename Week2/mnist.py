from einops.layers.torch import Rearrange, Reduce
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import torchvision



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

# class NNLayer(nn.Module):
#     def __init__(self, dim_in, dim_out) -> None:
#         super().__init__()
#         self.linear = nn.Linear(dim_in, dim_out)
#         self.act = nn.GELU()
    
#     def forward(self, x):
#         x = self.linear(x)
#         x = self.act(x)
#         return x

class NNLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.act = nn.GELU()
    def forward(self, x):
        shortcut = x
        x = self.act(self.linear(x))
        return shortcut + x

model = nn.Sequential(
    Rearrange("b c h w -> b (c h w)"),
    NNLayer(3 * 32 * 32, 256),
    NNLayer(256, 256),
    NNLayer(256, 256),
    NNLayer(256, 256),
    nn.Linear(256, 10)
).to(device)

print(f"{sum([np.prod(x.data.shape) for x in model.parameters()]):,}")

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(20):
    print(f"Starting epoch {epoch}...")
    correct = 0; total = 0
    pbar = tqdm(dataloader_train)
    for x, y in pbar:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()      # Zeroes things out
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        pbar.set_description(f"loss: {loss:.4f}")
        loss.backward()
        optimizer.step()
        
        correct += (y_hat.argmax(dim=1) == y).sum()
        total += y.shape[0]
    print(f"Training Acc: {correct/total*100:.4f}%")
        
    # acc = #correct / #total
    correct = 0
    total = 0
    for x, y in dataloader_test:
        x = x.to(device); y = y.to(device)
        y_hat = model(x)
        y_hat = y_hat.argmax(dim=1)
        correct += (y_hat == y).sum()
        total += y.shape[0]
    print(f"Acc: {correct/total*100:.4f}%")

torch.save(model.cpu(), 'model.pt')


# Before the training starts, print the total number of parameters. So a weight matrix 8 x 5 would have 40 parameters.
# For model, tally (sum up) the total num of parameters
# hint: `list(model.parameters())` gives you a list of all the parameters (weight & biases) so 


# ssh daa5724@euler.hbg.psu.edu -L 8888:localhost:8888
# time conda activate torch
# jupyter lab --no-browser --port=8888 