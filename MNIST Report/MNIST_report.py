from einops.layers.torch import Rearrange
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torchvision

device = "cuda"

dataset_train = torchvision.datasets.MNIST(".", train=True, download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307), (0.3081)),
]))
dataset_test = torchvision.datasets.MNIST(".", train=False, download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307), (0.3081)),
]))

dataloader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=4096)

class NNLayer(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        dim_1 = dim_in * 4
        dim_2 = dim_1 * 4
        self.linear = nn.Linear(dim_in, dim_1)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim_1, dim_2)
        self.act = nn.ReLU()
        self.linear3 = nn.Linear(dim_2, dim_out)
        self.residual = nn.Linear(dim_in, dim_out) if dim_in != dim_out else None

    def forward(self, x):
        org = self.linear(x)
        org = self.act(org)
        org = self.linear2(org)
        org = self.act(org)
        org = self.linear3(org)
        if self.residual is None:
            return x + org
        else:
            return self.residual(x) + org

model = nn.Sequential(
        Rearrange("b c h w -> b (c h w)"),
        NNLayer(1 * 28 * 28, 256),
        NNLayer(256, 256),
        NNLayer(256, 256),
        NNLayer(256, 256),
        nn.Linear(256, 10)
    ).to(device)

print(f"{sum([np.prod(x.data.shape) for x in model.parameters()]):,}")

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

for epoch in range(10):
    print(f"Starting epoch {epoch}...")
    correct = 0; total = 0
    pbar = tqdm(dataloader_train)
    for x, y in pbar:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()     
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        pbar.set_description(f"loss: {loss:.4f}")
        loss.backward()
        optimizer.step()
        
        correct += (y_hat.argmax(dim=1) == y).sum()
        total += y.shape[0]
    print(f"Training Acc: {correct/total*100:.4f}%")

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

