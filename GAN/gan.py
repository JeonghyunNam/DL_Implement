from shutil import register_unpack_format
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Maxout(nn.Module):
    def __init__(self, n_input, n_output, n_linear):
        super().__init__()
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_linear = n_linear
        self.layer = nn.Linear(n_input, n_output * n_linear)
        self.maxpool = nn.MaxPool1d(self.n_linear)
        self.reset_parameters()

    def reset_parameters(self):
        irange = 0.005
        nn.init.uniform_(self.layer.weight, -irange, irange)
        nn.init.uniform_(self.layer.bias, -irange, irange)

    def forward(self, input):
        intermediate = self.layer(input)
        output = F.max_pool1d(intermediate, kernel_size=self.n_linear)
        return output

class Generator(nn.Module):
    def __init__(self, n_input) -> None:
        super(Generator, self).__init__()
        self.n_input = n_input
        self.layer = nn.Sequential( nn.Linear(self.n_input, 1200),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(1200, 1200),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(1200, 784),
                                    nn.Sigmoid()  
                                  )
        for layer in self.layer:
            self.reset_parameters(layer)
        
    def reset_parameters(self, layer):
        irange = 0.05
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, -irange, irange)
            nn.init.uniform_(layer.bias, -irange, irange)

    def forward(self, input):
        output = self.layer(input)
        return output

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential( Maxout(n_input=784, n_output=240, n_linear=5),
                                    Maxout(n_input=240, n_output=240, n_linear=5),
                                    nn.Linear(240, 1),
                                    nn.Sigmoid()  
                                  )
        for layer in self.layer:
            if not isinstance(layer, Maxout):
                self.reset_parameters(layer)
        
    def reset_parameters(self, layer):
        irange = 0.005
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, -irange, irange)
            nn.init.uniform_(layer.bias, -irange, irange)

    def forward(self, input):
        output = self.layer(input)
        return output


def init_weights(m, irange):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -irange, irange)

if __name__ == '__main__':
    g = Generator(100)
    d = Discriminator()

    # test for Discriminator
    d_input = torch.randn(10,784)
    d_output = d(d_input)
    
    # test for Generator
    g_input = torch.randn(10,100)
    g_output = g(g_input)
    print(d)
    print(g)

