import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import datetime

# parameter
data_root = 'C:/Users/ys499/data'
result_root = 'C:/Users/ys499/Desktop/DL_implement/ALEXNET/Exp'
log_root = result_root + '/log'
ckpt_root = result_root + '/ckpt'

image_size = 227
num_workers = 2
num_classes = 1000
num_epoches = 90

input_size = 64
batch_size = 128

momentum = 0.9
weight_decay = 0.0005
learning_rate = 0.01
iter = 1

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # Convolution
        self.u_first_convlayer = nn.Sequential(
                                    nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 0),
                                    nn.ReLU(True),
                                    nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),
                                    nn.MaxPool2d(kernel_size= 3, stride= 2)
                                    )

        self.u_second_convlayer = nn.Sequential(
                                    nn.Conv2d(48, 128, kernel_size = 5, stride = 1, padding = 2),
                                    nn.ReLU(True),
                                    nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),
                                    nn.MaxPool2d(kernel_size= 3, stride= 2)
                                    )

        self.u_forth_convlayer = nn.Sequential(
                                    nn.Conv2d(192, 192, kernel_size = 3, stride = 1, padding = 1),
                                    nn.ReLU(True)
                                    )

        self.u_fifth_convlayer = nn.Sequential(
                                    nn.Conv2d(192, 128, kernel_size= 3, stride = 1, padding = 1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(kernel_size= 3, stride= 2)
                                    )

        self.l_first_convlayer = nn.Sequential(
                                    nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 0),
                                    nn.ReLU(True),
                                    nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),
                                    nn.MaxPool2d(kernel_size= 3, stride= 2)
                                    )

        self.l_second_convlayer = nn.Sequential(
                                    nn.Conv2d(48, 128, kernel_size = 5, stride = 1, padding = 2),
                                    nn.ReLU(True),
                                    nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),
                                    nn.MaxPool2d(kernel_size= 3, stride= 2)
                                    )

        self.l_forth_convlayer = nn.Sequential(
                                    nn.Conv2d(192, 192, kernel_size = 3, stride = 1, padding = 1),
                                    nn.ReLU(True)
                                    )
        self.l_fifth_convlayer = nn.Sequential(
                                    nn.Conv2d(192, 128, kernel_size= 3, stride = 1, padding = 1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(kernel_size= 3, stride= 2)
                                    )

        self.third_convlayer = nn.Sequential(
                                    nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),
                                    nn.ReLU(True)
                                    )   

        # Fully Connected
        self.fc_layer = nn.Sequential(
                                nn.Linear(6*6*256, 4096),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(4096,4096),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(4096, 1000),
                                nn.ReLU(True)
        )

    def forward(self, input):
        b_size = input.size(0)
        a_conv1 = self.u_first_convlayer(input)
        sa_conv1= a_conv1.split(48, dim=1)
        sa1_conv2, sa2_conv2 = self.u_second_convlayer(sa_conv1[0]), self.l_second_convlayer(sa_conv1[1])
        a_conv2 = torch.cat([sa1_conv2, sa2_conv2], dim=1)
        a_conv3 = self.third_convlayer(a_conv2)
        sa1_conv3, sa2_conv3 = a_conv3.split(192, dim=1)
        sa1_conv4, sa2_conv4 = self.u_forth_convlayer(sa1_conv3), self.l_forth_convlayer(sa2_conv3)
        sa1_conv5, sa2_conv5 = self.u_fifth_convlayer(sa1_conv4), self.l_fifth_convlayer(sa2_conv4)
        a_conv5 = torch.cat([sa1_conv5, sa2_conv5], dim = 1)
        result = self.fc_layer(a_conv5.view(b_size, -1))
        return result

def weights_init(m):
    classname = m.__class__.__name__   
    if (classname.find('Conv') != -1) and (m.in_channels == 48 or m.in_channels == 192):    # -1 = does not exist
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.constant_(m.bias.data, 1)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.constant_(m.bias.data, 1)
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.constant_(m.bias.data, 0)



if __name__ == '__main__':
    
    # File Specification
    d = datetime.datetime.now()

    os.makedirs(ckpt_root, exist_ok = True)
    os.makedirs(log_root, exist_ok = True)
    tbwriter = SummaryWriter(log_dir = log_root)
    
    # Random Seed
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # Data Loading
    dataset = dsets.CIFAR10(data_root, 
                            transform=transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        ]),
                                        download = True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size= batch_size, shuffle= True, num_workers=num_workers)
    
    # GPU specification
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #model generate
    net = AlexNet().to(device)
    net.apply(weights_init)
    print(net)

    # Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum= momentum, weight_decay= weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 30, gamma= 0.1)

    for epoch in range(num_epoches):

        for img, labels in dataloader:
            img , labels = img.to(device) , labels.to(device)
            
            # Calculate Loss
            output = net(img)
            loss = criterion(output, labels) 

            # Gradient Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(iter%10 == 0):
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == labels)
                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'.format(epoch + 1, iter, loss.item(), accuracy.item()))
                    tbwriter.add_scalar('Loss', loss.item(), iter)
                    tbwriter.add_scalar('Accuracy', accuracy.item(), iter)

            if(iter%100 == 0):
                with torch.no_grad():
                    print('*' * 10)
                    for name, parameter in net.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), iter)
                            tbwriter.add_histogram('grad/{}'.format(name), parameter.grad.cpu().numpy(), iter)
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print('\t{} - parameter_avg: {}'.format(name, avg_weight))
                            tbwriter.add_histogram('weight/{}'.format(name), parameter.data.cpu().numpy(), iter)
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), iter)
       
            iter+=1
            checkpoint_path = os.path.join(ckpt_root, 'alexnet_state_{}_{}.pkl'.format(epoch+1, str(d.strftime('%y%m%d%H%M'))))
            state = {
                'epoch': epoch,
                'iter': iter,
                'optimizer': optimizer.state_dict(),
                'model': net.state_dict(),
                'seed' : manualSeed,
            }
            torch.save(state, checkpoint_path)
                       
        lr_scheduler.step()         


