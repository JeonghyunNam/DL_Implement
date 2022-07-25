import gan
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import params as p
import data
from tqdm import tqdm
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

"""
    GAN model test
    Train/Validate part
"""
def lr_schduler(step):
    new_lr = 0.1 /(1.000004**step)
    return max(new_lr, 0.000001)

def momemtum_scheduler(epoch):
    if epoch <250:
        new_momentum = (0.2*epoch)/250 + 0.5
    else:
        new_momentum = 0.7
    return new_momentum

def main():

    # Dataset / Save
    datapath = p.datapath
    date = datetime.now().strftime('%m%d%H%M')
    savepath = p.savepath + date
    train_loader = data.make_train_datasets(100, datapath)
    writer = SummaryWriter(p.tbpath)
    
    # Model Gerneration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = gan.Generator(100).to(device)
    d = gan.Discriminator().to(device)

    # Parameter Specification

    n_epoch = 500
    step = 0
    init_lr = 0.1
    init_momentum = 0.5
    
    # Optimizer, Loss
    # criterion = nn.BCELoss()
    g_optim = optim.Adam(g.parameters(), lr = 0.0001, betas=(0.5,0.999))
    d_optim = optim.Adam(d.parameters(), lr = 0.0001, betas=(0.5,0.999))
    # g_optim = optim.SGD(g.parameters(), lr=init_lr, momentum=init_momentum)
    # d_optim = optim.SGD(d.parameters(), lr=init_lr, momentum=init_momentum)

    # Test noises
    test_noise = torch.FloatTensor(100,100).uniform_(-np.sqrt(3), np.sqrt(3)).to(device)
    # print(test_noise)

    for epoch in range(n_epoch):
        
        # Generating Part:
        d.train()
        g.train()
        G_avg_loss, D_avg_loss = 0, 0
        pbar1 = tqdm(enumerate(train_loader))
        for i, (image, _) in pbar1:

            # Discriminator
            d.zero_grad()
            n_batch = image.size(0)
            real_input = image.reshape(n_batch, -1).to(device)
            dfake_input = torch.FloatTensor(n_batch,100).uniform_(-np.sqrt(3), np.sqrt(3)).to(device)
            real_p     = d(real_input)
            fake_p     = d(g(dfake_input))

            real_loss  = -1 * torch.log(real_p)
            fake_loss  = -1 * torch.log(1.-fake_p)
            d_loss     = (real_loss+fake_loss).mean()
            D_avg_loss +=d_loss.item()

            d_loss.backward()
            d_optim.step()

            # Generator
            g.zero_grad()
            gfake_input = torch.FloatTensor(n_batch, 100).uniform_(-np.sqrt(3), np.sqrt(3)).to(device)
            g_p = d(g(gfake_input))
            g_loss = torch.log(1 - g_p).mean()
            G_avg_loss += g_loss.item()

            g_loss.backward()
            g_optim.step()
            
            step+=1

            # lr = lr_schduler(step)
            # d_optim.param_groups[0]['lr'] = lr
            # g_optim.param_groups[0]['lr'] = lr

            pbar1.set_postfix({'g_loss' : g_loss.item(), 'd_loss' : d_loss.item()})
        G_avg_loss/=i
        D_avg_loss/=i
        print(f"Epoch : {epoch}, G_loss : {G_avg_loss}, D_loss : {D_avg_loss}")

        # Confirming images
        with torch.no_grad():
            g.eval()
            d.eval()
            os.makedirs(name=savepath, exist_ok=True)
            test_image = g(test_noise).reshape(-1,1,28,28)
            vutils.save_image(test_image, "{0}/{1}.jpg".format(savepath, str(epoch+1).zfill(2)), normalize=True)
        
        # momentum =  momemtum_scheduler(epoch)
        # d_optim.param_groups[0]['momentum'] = momentum
        # g_optim.param_groups[0]['momentum'] = momentum
        # print(f"lr : {lr}, momentum : {momentum}")

        #TODO: g tensorboard (g_loss)
        writer.add_scalar('g_loss', G_avg_loss, epoch)

        #TODO: d tensorboard (d_loss)
        writer.add_scalar('d_loss', D_avg_loss, epoch)

        #TODO: epoch 10마다 저장 (epoch+1 % 10 == 0), 경로: model/GAN_{epoch+1}.pth
        if (((epoch+1) % 10 == 0) or (epoch == 0)):
            os.makedirs(p.savepath +'model/', exist_ok= True)
            torch.save({'g_state_dict': g.state_dict(),'d_state_dict': d.state_dict()}, p.savepath +'model/'+ (str)(epoch+1)+'.pth')

    writer.close()


if __name__ == "__main__":
    main()