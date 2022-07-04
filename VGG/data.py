import torch
import torch as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data

def make_train_datasets(batchsize, dataroot):
    train_dataset = dsets.ImageNet(root= dataroot,
            split= 'train',
            transform= transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
            ])
        )
    print("finished!!")
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= batchsize,
                        shuffle= True, num_workers=2)
    return dataloader

def make_val_datasets(batchsize, dataroot):
    test_dataset = dsets.ImageNet(root= dataroot,
            split='val',
            transform= transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
            ])
        )
    dataloader=  torch.utils.data.DataLoader(test_dataset, batch_size= batchsize, 
                        shuffle= True, num_workers=2)
    print("finished!!")
    return dataloader


if __name__ == '__main__':
    dataroot = "C:/Users/ys499/data/ImageNet/"
    make_val_datasets(256, dataroot)
    # make_train_datasets(256, dataroot)
    print("finished")
