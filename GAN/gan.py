import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Maxout(nn.Module):
    """Class Maxout implements maxout unit introduced in paper by Goodfellow et al, 2013.
    
    :param in_feature: Size of each input sample.
    :param out_feature: Size of each output sample.
    :param n_channels: The number of linear pieces used to make each maxout unit.
    :param bias: If set to False, the layer will not learn an additive bias.
    """
    
    def __init__(self, in_features, out_features, n_channels, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_channels = n_channels
        self.weight = nn.Parameter(torch.Tensor(n_channels * out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_channels * out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def forward(self, input):
        a = F.linear(input, self.weight, self.bias)
        b = F.max_pool1d(a.unsqueeze(-3), kernel_size=self.n_channels)
        return b.squeeze()

    def reset_parameters(self):
        irange = 0.005
        nn.init.uniform_(self.weight, -irange, irange)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -irange, irange)


class Generator(nn.Module):
    def __init__(self, n_input) -> None:
        super(Generator, self).__init__()
        self.n_input = n_input
        self.maxout1 = Maxout(784,240, 5)
        self.maxout2 = Maxout()
        self.layer = nn.Sequential( nn.Linear(self.n_input, 1200),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(1200, 1200),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(1200, 784),
                                    nn.Sigmoid()  
                                  )

    def forward(self, input):
        output = self.layer(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, n_batch) -> None:
        super(Discriminator, self).__init__()
        self.n_batch = n_batch
        self.layer = nn.Sequential( Maxout(n_channels=5, bias= True),
                                    nn.Linear(240, 240),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(240, 1),
                                    nn.Sigmoid()  
                                  )



def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -0.05, 0.05)

if __name__ == '__main__':
    # g = Generator(100)
    # g.apply(init_weights)
    # for name, param in g.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name} : {param.data}")
    # print(g)