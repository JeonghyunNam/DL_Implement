import torch 
import torch.nn as nn
import torchvision.utils as dutils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data

def make_train_datasets(batchsize = 100, dataroot = None):
    train_dataset = dsets.MNIST(root= dataroot,
            train= True,
            transform= transforms.Compose([ transforms.ToTensor(),
                                            # transforms.Normalize(mean = [0.5,], std  = [0.5,])
                                            ]),
        download= True)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= batchsize,
                        shuffle= True, num_workers=2)
    return dataloader

def make_val_datasets(batchsize = 100, dataroot = None):
    test_dataset = dsets.MNIST(root= dataroot,
            train= False,
            transform= transforms.Compose([ transforms.ToTensor(),
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
