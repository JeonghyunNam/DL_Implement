import torch
import torch.nn as nn
import torchvision.transforms as transforms


class UNet(nn.Module):
    def __init__(self) -> None:
        super(UNet, self).__init__()
        self.downlayer1 = self.constructDown(1, 64)
        self.downlayer2 = self.constructDown(64, 128)
        self.downlayer3 = self.constructDown(128, 256)
        self.downlayer4 = self.constructDown(256, 512)
        self.midlayer   = self.constructUp(512, 1024)
        self.uplayer1   = self.constructUp(1024, 512)
        self.uplayer2   = self.constructUp(512, 256)
        self.uplayer3   = self.constructUp(256, 128)
        self.final      = self.constructFinal()

    def constructDown(self, c_input_size, c_output_size):
        layerlist = []
        if c_input_size != 1:
            layerlist += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layerlist += [ nn.Conv2d(c_input_size, c_output_size, kernel_size=3, stride=1, padding=0),
                       nn.ReLU(inplace = True),
                       nn.Conv2d(c_output_size, c_output_size, kernel_size=3, stride=1, padding=0),
                       nn.ReLU(inplace= True)
                     ]
        return nn.Sequential(*layerlist)
    
    def constructUp(self, c_input_size, c_output_size):
        layerlist = []
        if c_input_size < c_output_size:
            layerlist += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layerlist += [nn.Conv2d(c_input_size, c_output_size, kernel_size=3, stride=1, padding=0),
                     nn.ReLU(inplace = True),
                     nn.Conv2d(c_output_size, c_output_size, kernel_size=3, stride=1, padding=0),
                     nn.ReLU(inplace= True),
                     nn.ConvTranspose2d(c_output_size, c_output_size//2, kernel_size=2, stride=2)
                    ]
        return nn.Sequential(*layerlist)
    
    def constructFinal(self):
        layerlist = [ nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
                      nn.ReLU(inplace = True),
                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                      nn.ReLU(inplace = True),
                      nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)
                    ]
        return nn.Sequential(*layerlist)

    def cropNconcat(self, input, target):
        o_wh = target.size(2)
        transform = transforms.CenterCrop(o_wh)
        cropped = transform(input)
        concated = torch.cat([cropped, target], dim=1)
        return concated

    def forward(self, input):
        down1 = self.downlayer1(input)
        down2 = self.downlayer2(down1)
        down3 = self.downlayer3(down2)
        down4 = self.downlayer4(down3)
        middle = self.midlayer(down4)

        up0 = self.cropNconcat(down4, middle)
        up1 = self.cropNconcat(down3,self.uplayer1(up0))
        up2 = self.cropNconcat(down2,self.uplayer2(up1))
        up3 = self.cropNconcat(down1,self.uplayer3(up2))
        output = self.final(up3)

        return output

if __name__ == '__main__':
    unet = UNet()
    
    # Testing model architecture
    input = torch.zeros(1,1, 572, 572)
    output = unet(input)
    print(output.size())
