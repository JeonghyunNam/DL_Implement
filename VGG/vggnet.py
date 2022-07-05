import torch
import torch.nn as nn
import torchvision.utils

import math
# import

class VGGNet(nn.Module):
    def __init__(self) -> None:
        super(VGGNet, self).__init__()
        self.module = []
        self.module = self.make_Conv(self.module)

        self.convlayer = nn.Sequential(*self.module)
        self.linlayer = nn.Sequential(nn.Linear(512, 4096),
                                      nn.ReLU(True),
                                      nn.Linear(4096, 4096),
                                      nn.ReLU(True),
                                      nn.Linear(4096, 100),
                                      )

    def make_Conv(self, module):
        _size = [3, 64, 128, 256, 512, 512]
        n_layers = [2, 2, 3, 3, 3]
        for i in range(len(_size)-1):
            outputsize = _size[i+1]
            inputsize = _size[i]
            module = self.make_block(module, inputsize, outputsize, n_layers[i])
        return module

    def make_block(self, module, inputsize, outputsize, n_layers):
        module.append(nn.Conv2d(inputsize, outputsize, kernel_size= 3, stride= 1, padding= 1))
        module.append(nn.ReLU(True))
        for i in range(n_layers-1):
            module.append(nn.Conv2d(outputsize, outputsize, kernel_size=3, stride=1, padding = 1))
            module.append(nn.ReLU(True))
        module.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return module
        
    def forward(self, x):
        x = self.convlayer(x)
        x = x.reshape(x.shape[0], -1)
        # print(x.size())
        x = self.linlayer(x)
        return x

if __name__ == '__main__':
    model = VGGNet()
    
