import torch
import torch as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data

def make_train_datasets(batchsize, dataroot):
    train_dataset = dsets.CIFAR10(root= dataroot,
            train= True,
            transform= transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
            ],
            ),download= True
            )
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= batchsize,
                        shuffle= True, num_workers=2)
    return dataloader

def make_test_datasets(batchsize, dataroot):
    test_dataset = dsets.CIFAR10(root= dataroot,
            train= False,
            transform= transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
            ],
            ),download= True)
    dataloader=  torch.utils.data.DataLoader(test_dataset, batch_size= batchsize, 
                        shuffle= True, num_workers=2)
    return dataloader