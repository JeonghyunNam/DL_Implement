import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.utils.data as d
from PIL import Image 
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import Augmentor

class CustomDataset(d.Dataset):
    def __init__(self, path, train = True):
        self.path = path    # Input path until train/val
        self.img_path = path +'/Image/'
        self.label_path = path + '/Label/'
        self.train = train
        self.img_list = [self.img_path+k for k in os.listdir(self.img_path)]
        self.label_list = [self.label_path + k for k in os.listdir(self.label_path)]

    def __len__(self):
        return len(self.img_list)
    
    
    def __transform__(self, image, mask):
        """
            Image Augmentation
            Have to be changed, but works well
        """
        # Random horizontal flipping
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)
        
        # Random rotating (have to be chaged)
        # if random.random() > 0.5:
        #     deg = random.randint(0, 10)
        #     image = F.rotate(image, deg, interpolation=transforms.InterpolationMode.BILINEAR)
        #     mask = F.rotate(mask, deg, interpolation=transforms.InterpolationMode.BILINEAR)
        
        # Elastic distortion (have to be changed)
        # if random.random() > 0.5:
        #     image.gaussian_distortion(probability=1, grid_width=3, grid_height=3, magnitude=5, corner='bell', method='in')
        return image, mask


    def __getitem__(self, idx):
        """
            Convert image, label to Tensor datatype
        """
        img_path = self.img_list[idx]
        label_path = self.label_list[idx]
        img, label = Image.open(img_path), Image.open(label_path)
        if self.train:
            img, label = self.__transform__(img, label)
        
        # Transform to tensor
        img = F.to_tensor(img)
        label = F.to_tensor(label)
        
        return img, label

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
    
    
