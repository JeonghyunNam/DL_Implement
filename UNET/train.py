import enum
from pickletools import optimize
from random import sample

import unet as u
import params as p
import data
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as dsets
import math
from skimage.measure import label
import scipy.ndimage.morphology as mor

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def calWmap(mask):
    # return label Wamp as tensor
    ## cal w_c
    mask = mask.cpu().numpy()
    pixel_map = np.unique(mask)

    w_map = []
    for i in range(len(pixel_map)):
        n_pixel = np.count_nonzero(mask == pixel_map[i])
        w_map.append(1/n_pixel)

    maximum = max(w_map)
    nw_map = [i / maximum for i in w_map]
    w_c = np.zeros((mask.shape))

    for i in range(len(pixel_map)):
        w_c[mask == pixel_map[i]] = nw_map[i]

    cells = label(mask, connectivity=2)
    bwgt = np.zeros(mask.shape)

    maps = np.zeros((mask.shape[1], mask.shape[2], np.amax(cells)))
    if np.amax(cells) >= 2:
        for ci in range(np.amax(cells)):
            maps[:,:,ci] = mor.distance_transform_edt(cells== ci)
        maps = np.sort(maps, axis=0)
        d1 = maps[:,:,0]
        d2 = maps[:,:,1]

        bwgt = 10*np.exp(-(np.multiply((d1+d2),(d1+d2))/50))

    w = w_c + bwgt
    wmax = np.amax(w)
    w /= wmax
    # print(w)
    return w

def CustomLoss(output, label):

    batch_size = output.size(0)
    loss = 0
    for i in range(batch_size):
        w = torch.tensor(calWmap(label[i])).to('cuda:0')
        loss += w.mul((torch.log(output[i][0]).mul(1-label))+(torch.log(output[i][1]).mul(label))).sum()
    # print(f"output size: {output.size()}")
    # print(f"output size: {output[0].size()}")
    # print(loss.size())
    # find l(x)
    return loss/batch_size


def main():
    ## Dataset prepare
    train_dataset = data.CustomDataset(p.train_path, train= True)

    val_dataset = data.CustomDataset(p.val_path, train= False)

    train_loader = dsets.DataLoader(dataset=train_dataset, batch_size=12, shuffle=True, drop_last=False)
    val_loader = dsets.DataLoader(dataset=val_dataset, batch_size= 12, shuffle=True, drop_last=False)

    writer = SummaryWriter(p.save_path)
    

    ## GPU Specification
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = u.UNet().to(device)
    model = nn.DataParallel(model, output_device=0)
    # model.apply(u.weights_init)
    criterion = CustomLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999))

    # preval_error = 0
    for epoch in range(p.max_epoch):
        print(f"Epoch: {epoch}")
        
        # train
        model.train()
        train_loss = 0
        pbar1 = tqdm(enumerate(train_loader))
        for i, (image, label) in pbar1:
            optimizer.zero_grad()
            label = label[:,:,92:92+388,92:92+388]
            image , label = image.to(device), label.to(device)
            output = model(image)

            # HAVE To Implement LOSS Term!!!
            
            loss = criterion(output, label)
            # print(loss.size())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar1.set_postfix({'loss': train_loss})
        # train_loss /= i
        
        # correct = 0
        # samples = 0
        # # test
        # with torch.no_grad():
        #     model.eval()
        #     val_loss = 0
        #     pbar2 = tqdm(enumerate(val_loader))
        #     for i, (image, label) in pbar2:
        #         image , label = image.to(device), label.to(device)
        #         output = model(image)
        #         loss = criterion(output, label)
        #         val_loss += loss.item()
        #         _, predict = torch.max(output.data, 1)
        #         correct += (predict == label).sum().item()
        #         samples += predict.size(0)      

        #         pbar2.set_postfix({'loss': loss})
        #     val_loss /= i
        # currval_error = 100.*correct/samples

        # if abs(currval_error - preval_error)<0.01:
        #     optimizer.param_groups[0]['lr']/=10
        #     print(optimizer.param_groups[0]['lr'])

        # preval_error = currval_error

        # print(f"Train Loss: {train_loss}\nValidation Loss: {val_loss}")
        # print(f"Accuracy: {currval_error}")
            
        # #TODO: train tensorboard (train_loss)
        # writer.add_scalar('train_loss', train_loss, epoch)

        # #TODO: valid tensorboard (val_loss)
        # writer.add_scalar('valid_loss', val_loss, epoch)

        # #TODO: accuracy tensorboard (currval_error)
        # writer.add_scalar('accuracy (%)', currval_error, epoch)

        # #TODO: epoch 10마다 저장 (epoch+1 % 10 == 0), 경로: model/VGG_{epoch+1}.pth
        # if (((epoch+1) % 10 )== 0 or (epoch == 0)):
        #     os.makedirs(p.save_path +'model/', exist_ok= True)
        #     torch.save(model.state_dict(), p.save_path +'model/'+ (str)(epoch+1)+'.pth')

    writer.close()


if __name__ == "__main__":
    print("Main")
    main()
    # input = torch.randint(0,2, (1,1,388,388))
    # calWmap(input)
