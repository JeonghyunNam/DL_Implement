import random
import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torchsummary import summary as summary

class Alexnet(nn.Module):
    def __init__(self) -> None:
        super(Alexnet, self).__init__()
        # Conv
        self.block1 = nn.Sequential(
                        nn.Conv2d(3, 96, kernel_size=11, stride= 4, padding= 0), #(227 - 11)/4 + 1 = 55
                        nn.ReLU(True),
                        nn.LocalResponseNorm(size = 5, alpha= 0.0001, beta= 0.75, k= 2),
                        nn.MaxPool2d(kernel_size= 3, stride= 2) #27
                    )
        self.block2 = nn.Sequential(
                        nn.Conv2d(96, 256, kernel_size= 5, stride= 1, padding= 2),   #(27+2*2-5)/1 + 1 = 27
                        nn.ReLU(True),
                        nn.LocalResponseNorm(size = 5, alpha= 0.0001, beta= 0.75, k= 2),
                        nn.MaxPool2d(kernel_size= 3, stride= 2) #13
                    )
        self.block3 = nn.Sequential(
                        nn.Conv2d(256, 384, kernel_size= 3, stride= 1, padding= 1),  #(13+2*1 -3)/1 +1 = 13
                        nn.ReLU(True)
                    )
        self.block4 = nn.Sequential(
                        nn.Conv2d(384, 384, kernel_size= 3, stride= 1, padding= 1),  #(13+2*1 -3)/1 +1 = 13
                        nn.ReLU(True)
                    )
        self.block5 = nn.Sequential(
                        nn.Conv2d(384, 256, kernel_size= 3, stride= 1, padding= 1),  #(13+2*1 -3)/1 +1 = 13
                        nn.ReLU(True),
                        nn.MaxPool2d(kernel_size= 3, stride= 2)
                    )
        # FC
        self.dense1= nn.Sequential(
                        nn.Linear(256*6*6,4096),
                        nn.ReLU(True),
                        nn.Dropout(0.5)
                    )
        self.dense2= nn.Sequential(
                        nn.Linear(4096,4096),
                        nn.ReLU(True),
                        nn.Dropout(0.5)
                    )
        self.dense3= nn.Sequential(
                        nn.Linear(4096, 10),
                        # nn.ReLU(True)
                    )
        
        self._init_weight()

    def _init_weight(self):
        for layername, child in self.named_children():
            if isinstance(child, nn.Sequential):
                sublayername = layername
                if sublayername == 'block1' or sublayername == 'block3':
                    for _, schild in child.named_children():
                        if isinstance(schild, nn.Conv2d):
                            nn.init.normal_(schild.weight.data, mean=0, std=0.01)
                            nn.init.constant_(schild.bias.data, 0)
                else:
                    for _, schild in child.named_children():
                        if isinstance(schild, nn.Conv2d)or isinstance(schild, nn.Linear) :
                            nn.init.normal_(schild.weight.data, mean=0, std=0.01)
                            nn.init.constant_(schild.bias.data, 1)

                            
    def forward(self, input):
        batch_size = input.size(0)
        output = self.block1(input)
        output = self.block2(output)
        output = self.block3(output)
        # print(output.size())
        output = self.block4(output)
        # print(output.size())
        output = self.block5(output)
        # print(output.size())
        output = output.view(-1, 256*6*6)
        output = self.dense1(output)
        # print(output.size())
        output = self.dense2(output)
        # print(output.size())
        output = self.dense3(output)
        
        return output
        

if __name__ == '__main__':
    device = 'cuda:0'
    alexnet = Alexnet().to(device)
    # print(alexnet)
    # summary(alexnet, (3,227,227))