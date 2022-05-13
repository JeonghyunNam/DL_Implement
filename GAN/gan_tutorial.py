import os
import random
from turtle import forward
from matplotlib.pyplot import imshow
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import torch.optim as optim

import matplotlib.pyplot as plt


# random seed
randomSeed = 999
random.seed(randomSeed)
torch.manual_seed(randomSeed)

# hyperparameter
dataroot = 'C:/Users/ys499/data/MNIST'
ckptroot = './ckptroot'
workers = 2
batch_size = 64
n_epoch = 10
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
ngf = 64
ndf = 64
d_noise = 100
nc = 1
img_size = 28

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(d_noise, ngf),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(ngf, ngf),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(ngf, img_size*img_size),
                nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                nn.Linear(img_size*img_size, ndf),
                nn.LeakyReLU(0.2, inplace= True),
                nn.Dropout(0.1),
                nn.Linear(ndf, ndf),
                nn.LeakyReLU(0.2, inplace= True),
                nn.Dropout(0.1),
                nn.Linear(ndf, 1),
                nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.model(input)

if __name__ =='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = datasets.MNIST(root= dataroot, 
            transform=transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,),(0.5,))
            ]),
    )

    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size= batch_size,
                shuffle=True,
                num_workers=workers
    )

    criterion =  nn.BCELoss()
    G = Generator().to(device)
    D = Discriminator().to(device)

    optimG = optim.Adam(G.parameters(), lr = lr, betas=(beta1, beta2))
    optimD = optim.Adam(D.parameters(), lr = lr, betas=(beta1, beta2))

    fixed_noise = torch.randn(batch_size, d_noise, device=device)

    real_label = 1.
    fake_label = 0.

    img_list = []
    G_losses = []
    D_losses = []
    iter = 0

    os.makedirs(ckptroot, exist_ok=True)
    print('Learning Start')

    for epoch in range(n_epoch):
        for i, data in enumerate(dataloader, 0):
            #   Discriminator learning part
            #   real_sample
            real_sample = data[0].to(device)
            b_size = real_sample.size(0)
            label = torch.full((b_size, ), real_label, dtype=torch.float, device=device)
            D.zero_grad()
            out = D(real_sample.view(b_size, -1)).view(-1)

            errD_real = criterion(out, label)
            errD_real.backward()
            D_x = out.mean().item()

            # fake_sample
            noise = torch.randn(b_size, d_noise, device = device)
            fake_sample = G(noise)
            label.fill_(fake_label)
            out = D(fake_sample.detach()).view(-1)
            first = True
            if first:
                # print(fake_sample.size())
                # print(real_sample.size())
                first = False

            if (i%100 == 0):
                fake_sample = fake_sample.reshape(-1, nc, img_size, img_size)
                vutils.save_image(fake_sample, "./{0}/{1}_{2}.jpg".format(ckptroot, str(epoch+1).zfill(2), str(i).zfill(5)), normalize=True)
                # vutils.save_image(real_sample, "./{0}/{1}_{2}.jpg".format(ckptroot, str(epoch).zfill(2), str(i).zfill(5)), normalize=True)
            errD_fake = criterion(out, label)
            errD_fake.backward()
            D_G_z = out.mean().item()
            errD = 0.5*(errD_real + errD_fake)
            
            # Gradient update
            optimD.step()

            #   Generator learning part
            G.zero_grad()
            label.fill_(real_label)
            fake_sample = fake_sample.reshape(-1, img_size*img_size)
            out = D(fake_sample).view(-1)
            errG = criterion(out, label)
            errG.backward()
            G_z = out.mean().item()
            optimG.step()

            if i%50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch+1, n_epoch, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z, G_z))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iter%500 == 0) or ((epoch == n_epoch-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = G(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding =2, normalize=True))
            
            iter +=1

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

    real_batch = next(iter(dataloader))

    # Plot the fake images from the last epoch
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()
  

