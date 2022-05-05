############################################################################
#   Goal            :      Implement dcgan structure described in paper    #
#   Direction       : Modularization                                       #
#   Acknowledgement : Sunghyeon Kim                                        #
############################################################################

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.functional as F

import torch.optim as optim

import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import matplotlib.animation as anim
from IPython.display import HTML

import os

## Radom seed
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

## Parameters
dataroot = 'C:/Users/ys499/data/celeba'           # data directory
ckptroot = "./checkpoint"
workers = 2             # # of thread
batch_size = 128        # # of mini_batch
image_size = 64         # output of generater 3*64*64
nc = 3                  # dim of output channel
nz = 100                # dim of input noise
ngf = 128               # dim of generator feature
ndf = 128                # dim of discriminator feature
num_epochs = 5          # # of epoch
lr = 0.0002             # learning rate
beta1 = 0.5             # coeff of Adam optim
ngpu = 1                # # of gpu operating

## Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.proj = nn.Parameter(torch.normal(0, 0.02, size = (1, nz, ngf*8*4*4)))
        self.proj.requires_grad = True
        self.batchnorm = nn.BatchNorm2d(ngf*8)
        self.relu = nn.ReLU(True)

        self.main = nn.Sequential(

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        # inter = torch.tensordot(input, self.proj, dims= ([1,2,3],[0,3,2]))
        b_size = input.size(0)
        input = input.reshape(b_size, 1, nz)
        inter = torch.bmm(input, self.proj.repeat(b_size, 1, 1))
        inter = self.batchnorm(inter.reshape([b_size, ngf*8, 4 ,4]))
        inter = self.relu(inter)
        inter = inter.reshape([b_size, ngf*8, 4 , 4])
        return self.main(inter)

## Discriminator Class
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.proj = nn.Parameter(torch.normal(0, 0.02, size=(1, ngf*8*4*4, 1)))
        self.sigmoid = nn.Sigmoid()
        self.proj.requires_grad = True
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf *4),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace = True),
        )

    def forward(self, input):
        b_size = input.size(0)
        inter = self.main(input)
        # inter = torch.matmul(torch.reshape(inter, (-1,ngf*8*4*4)),self.proj.reshape([ngf*8*4*4, -1]))
        inter = inter.reshape(b_size, 1, -1)
        inter = torch.bmm(inter, self.proj.repeat(b_size, 1, 1))
        return inter

## Weight Initialization
def weights_init(m):
    classname = m.__class__.__name__    # reference method of getting classname
    if classname.find('Conv') != -1:    # -1 = does not exist
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
## main part

if __name__ == '__main__':
    ## Data preparation
    dataset = dset.ImageFolder(root = dataroot,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,
                            shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0 ) else "cpu")

    real_batch = next(iter(dataloader))
    plt.figure(figsize = (8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
                    padding=2, normalize=True).cpu(), (1, 2, 0)))

    # declare Generator, Discriminator
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netG.apply(weights_init)
    netD.apply(weights_init)
    #print(list(netG.parameters()))
    #print(list(netD.parameters()))
    print(netG)
    print(netD)

    # Loss Function & Optimizer
    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(128, nz, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr= lr * 0.8, betas = (beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr= lr, betas = (beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    os.makedirs(ckptroot, exist_ok=True)

    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device = device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = (torch.rand(b_size, nz, 1, 1, device = device)*2)-1
            fake = netG(noise)
            if i%100 == 0:
                # vutils.save_image(real_cpu, "./{0}/real_{1}.jpg".format( ckptroot, str(i).zfill(5), normalize=True, value_range=(-1,1)))
                vutils.save_image(fake, "./{0}/{1}_{2}.jpg".format( ckptroot, str(epoch).zfill(2), str(i).zfill(5), normalize=True, value_range=(-1,1) ))
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i%50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch+1, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters%500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding =2, normalize=True))
            
            iters +=1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label= "G")
    plt.plot(D_losses, label = "D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]

    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()