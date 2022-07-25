import enum
from pickletools import optimize
from random import sample

import unet as u
import params as p
import data
import numpy as np
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as dsets
import torchvision.utils as vutils
import math
from skimage.measure import label
import scipy.ndimage.morphology as mor

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings('ignore')

def CustomLoss(input, label, w):
    """
        Implementation of Pixelwise weighted crossentropy loss 
    """
    output = F.binary_cross_entropy(input=F.softmax(input, dim=1)[:,1,:,:], target=label, weight=w)
    # print(f"output size: {output.size()}")
    # print(f"output size: {output[0].size()}")
    # print(loss.size())
    return output


def main():
    ## Dataset prepare
    datestr = datetime.now().strftime('%m%d%H%M')
    os.makedirs(p.save_path + datestr+'/', exist_ok=True)
    train_dataset = data.CustomDataset(p.train_path, train= True)

    val_dataset = data.CustomDataset(p.val_path, train= False)

    train_loader = dsets.DataLoader(dataset=train_dataset, batch_size=12, shuffle=True, drop_last=False)
    val_loader = dsets.DataLoader(dataset=val_dataset, batch_size=12, shuffle=True, drop_last=False)

    writer = SummaryWriter(p.save_path)
    

    ## GPU Specification
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = u.UNet().to(device)
    model = nn.DataParallel(model, output_device=0)
    # model.apply(u.weights_init)
    criterion = CustomLoss
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999))

    for epoch in range(p.max_epoch):
        print(f"Epoch: {epoch}")
        
        # train
        model.train()
        train_loss = 0
        pbar1 = tqdm(enumerate(train_loader))
        for i, (image, label, weight) in pbar1:
            optimizer.zero_grad()
            label = label[:,:,92:92+388,92:92+388].float()
            weight = weight[:,:,92:92+388,92:92+388].float()
            image , label = image.to(device), label.reshape(-1, 388, 388).to(device)
            weight = weight.reshape(-1, 388, 388).to(device)
            output = model(image)
            loss = criterion(output, label, weight)
            # print(loss.size())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar1.set_postfix({'loss': loss.item()})
        train_loss /= i
        
        # test
        with torch.no_grad():
            model.eval()
            val_loss = 0
            pbar2 = tqdm(enumerate(val_loader))
            for i, (image, label, weight) in pbar2:
                label = label[:,:,92:92+388,92:92+388].float()
                weight = weight[:,:,92:92+388,92:92+388].float()
                image , label = image.to(device), label.reshape(-1,388,388).to(device)
                weight = weight.reshape(-1, 388, 388).to(device)
                output = model(image)
                loss = criterion(output, label, weight)
                val_loss += loss.item()
                # output = torch.as_tensor((output[0,1,:,:]-0.5)>0,)
                vutils.save_image(output[0,1:,:], "{0}{1}/{2}.jpg".format(p.save_path,datestr, str(epoch).zfill(2), normalize=True, value_range=[0,1]))   
                pbar2.set_postfix({'loss': loss.item()})
            val_loss /= i if i != 0 else val_loss

        print(f"Train Loss: {train_loss}\nValidation Loss: {val_loss}")

        # #TODO: train tensorboard (train_loss)
        writer.add_scalar('train_loss', train_loss, epoch)

        # #TODO: valid tensorboard (val_loss)
        writer.add_scalar('valid_loss', val_loss, epoch)

        # #TODO: epoch 10마다 저장 (epoch+1 % 10 == 0), 경로: model/VGG_{epoch+1}.pth
        if (((epoch+1) % 10 )== 0 or (epoch == 0)):
            os.makedirs(p.save_path +'model/', exist_ok= True)
            torch.save(model.state_dict(), p.save_path +'model/'+ (str)(epoch+1)+'.pth')

    writer.close()


if __name__ == "__main__":
    print("Main")
    main()
