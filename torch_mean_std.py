import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
#from tqdm.notebook import tqdm
#from time import time

N_CHANNELS = 3

#dataset = datasets.MNIST("data", download=True,
#                 train=True, transform=transforms.ToTensor())
#train_path=r"./Datasets/External_Orchid_Datasets/train-en"
train_path=r"./Datasets/training/train"


resize = 224

dataset = datasets.ImageFolder(train_path, transform=transforms.Compose([
        transforms.Resize(size=(resize,resize)),#(h,w)
        transforms.ToTensor()]))

image_data_loader = DataLoader(
  dataset, 
  # batch size is whole datset
  batch_size=len(dataset), 
  shuffle=False, 
  num_workers=0)

def mean_std(loader):
  images, lebels = next(iter(loader))
  # shape of images = [b,c,w,h]
  mean, std = images.mean([0,2,3]), images.std([0,2,3])
  return mean, std

mean, std = mean_std(image_data_loader)
print("mean and std: \n", mean, std)


dataset = datasets.ImageFolder(train_path, transform=transforms.Compose([
        transforms.Resize(size=(resize,resize)),#(h,w)
        transforms.ToTensor()]))

loader = torch.utils.data.DataLoader(dataset,
                         batch_size=10,
                         num_workers=0,
                         shuffle=False)

mean = 0.0
for images, _ in loader:
    batch_samples = images.size(0) 
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
mean = mean / len(loader.dataset)

var = 0.0
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    var += ((images - mean.unsqueeze(1))**2).sum([0,2])
std = torch.sqrt(var / (len(loader.dataset)*resize*resize))

print("mean and std: \n", mean, std)
