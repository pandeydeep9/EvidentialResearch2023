import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_train = MNIST("./data/mnist",
                   download=True,
                   train=True,
                   transform=transforms.Compose([transforms.ToTensor()]))


dataloader_train = DataLoader(
    data_train, batch_size=4, shuffle=True, num_workers=8)
