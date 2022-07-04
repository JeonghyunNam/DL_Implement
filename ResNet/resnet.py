
import torch
import torch.nn as nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride) -> None:
        super(ResBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        self.residual = nn.Sequential(
                            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride = self.stride, padding = 1),
                            nn.BatchNorm2d(self.out_channel),
                            nn.ReLU(True),
                            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding =1),
                            nn.BatchNorm2d(self.out_channel)
        )

        self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_channel != self.out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, stride= self.stride),
                nn.BatchNorm2d(self.out_channel)
            )
    def forward(self, x):
        return nn.ReLU(True)(self.residual(x)+ self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, resblock) -> None:
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding = 1, stride=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
        )
        self.conv2 = self.make_layer(resblock, 16, 16, 1)
        self.conv3 = self.make_layer(resblock, 16, 32, 2)
        self.conv4 = self.make_layer(resblock, 32, 64, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(8*8, 100)

    def make_layer(self, resblock,  in_channel, out_channel, stride):
        layers = []
        strides = [stride, 1, 1]
        for _stride in strides:
            layers.append(resblock(in_channel, out_channel, _stride))
            in_channel = out_channel

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        # print(f"after_conve1, {output.size()}")
        output = self.conv2(output)
        # print(f"after_conve2, {output.size()}")
        output = self.conv3(output)
        # print(f"after_conve3, {output.size()}")
        output = self.conv4(output)
        # print(f"after_conve4, {output.size()}")
        output = self.avg_pool(output)
        # print(f"after_avg_pool, {output.size()}")
        output = output.reshape(output.shape[0], -1)
        output = self.fc(output)

        return output

if __name__ == '__main__':

    resnet = ResNet(ResBlock)
    input = torch.rand(256,3,32,32)
    output = resnet(input)
    print(output)