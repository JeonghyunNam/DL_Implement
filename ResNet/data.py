import torch
import torch as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data
# ###########################################
# have to check datapath in param.py
# ###########################################

def make_train_datasets(batchsize, dataroot):
    train_dataset = dsets.CIFAR100(root= dataroot,
            train= True,
            transform= transforms.Compose([transforms.RandomResizedCrop(32),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.5,], std  = [0.5,])
                                            ]),
        download= True)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= batchsize,
                        shuffle= True, num_workers=2)
    return dataloader

def make_val_datasets(batchsize, dataroot):
    test_dataset = dsets.CIFAR100(root= dataroot,
            train= False,
            transform= transforms.Compose([transforms.RandomResizedCrop(32),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.5,], std  = [0.5,])
                                            ]),
        download=True)
    dataloader=  torch.utils.data.DataLoader(test_dataset, batch_size= batchsize, 
                        shuffle= True, num_workers=2)

    return dataloader


if __name__ == '__main__':
    ## you need to configure this path
    dataroot = "C:/Users/ys499/data"
    make_val_datasets(256, dataroot)
    make_train_datasets(256, dataroot)
    print("finished")
