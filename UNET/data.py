import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.utils.data as d
from PIL import Image 
import os
import matplotlib.pyplot as plt
import numpy as np
import random

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class CustomDataset(d.Dataset):
    def __init__(self, path, train = True):
        self.path = path    # Input path until train/val
        self.img_path = path +'/Image/'
        self.label_path = path + '/label/'
        self.weight_path = path +'/weights/'
        self.train = train
        self.img_list = [self.img_path+k for k in os.listdir(self.img_path)]
        self.label_list = [self.label_path + k for k in os.listdir(self.label_path)]
        self.weight_list = [self.weight_path + k for k in os.listdir(self.weight_path)]

    def __len__(self):
        return len(self.img_list)
    
    
    def __transform__(self, image, mask, weight):
        """
            Image Augmentation
            Have to be changed, but works well
        """
        # Random horizontal flipping
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
            weight = F.hflip(weight)

        # Random vertical flipping
        if random.random() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)
            weight = F.vflip(weight)
        
        # Random rotating (have to be chaged)
        if random.random() > 0.5:
            deg = random.randint(0, 10)
            image = F.rotate(image, deg, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = F.rotate(mask, deg, interpolation=transforms.InterpolationMode.BILINEAR)
            weight = F.rotate(weight, deg, interpolation=transforms.InterpolationMode.BILINEAR)
        
        # Elastic distortion (have to be changed)
        # if random.random() > 0.5:
        #     image.gaussian_distortion(probability=1, grid_width=3, grid_height=3, magnitude=5, corner='bell', method='in')
        
        
        
        return image, mask, weight


    def __getitem__(self, idx):
        """
            Convert image, label to Tensor datatype
        """
        img_path = self.img_list[idx]
        label_path = self.label_list[idx]
        weight_path = self.weight_list[idx]
        img, label, weight = Image.open(img_path), Image.open(label_path), Image.open(weight_path)
        if self.train:
            img, label, weight = self.__transform__(img, label, weight)
        
        # Transform to tensor
        img = F.to_tensor(img)
        label = F.to_tensor(label)
        weight = F.to_tensor(weight)
        
        return img, label, weight

if __name__ == '__main__':
    """
        Confirm CustomDataset works well 
    """
    dataset = CustomDataset('C:/Users/ys499/Desktop/DL_implement/UNET/data/train')
    dataloader = d.DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=False)
    sample = []
    for epoch in range(2):
        print(f"epoch : {epoch} ")
        for batch in dataloader:
            img, label = batch
            
            sample.append(img)
            sample.append(label)

    # plt.figure()
    figure, axis = plt.subplots(2,2)
    axis[0][0].imshow(np.transpose(sample[0][0],(1,2,0)), cmap = 'gray')
    axis[0][1].imshow(np.transpose(sample[1][0],(1,2,0)), cmap = 'gray')
    axis[1][0].imshow(np.transpose(sample[2][0],(1,2,0)), cmap = 'gray')
    axis[1][1].imshow(np.transpose(sample[3][0],(1,2,0)), cmap = 'gray')
    plt.show()
    
    
