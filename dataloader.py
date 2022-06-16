import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


def MNIST_load():
    batch_size_train = 256
    batch_size_test = 1000
    random_seed = 42
    torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                   ])),
        batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                   ])),
        batch_size=batch_size_test, shuffle=False)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print("Load data success!")
    return train_loader, test_loader


def main():
    train_loader = MNIST_load()
    print(train_loader[0])


if __name__ == '__main__':
    main()
