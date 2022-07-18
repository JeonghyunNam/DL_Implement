import torch
import torchvision.transforms as transforms
import torch.utils.data as d
from PIL import Image 
import os
import matplotlib.pyplot as plt
import numpy as np

class CustomDataset(d.Dataset):
    def __init__(self, path, transform=None):
        self.path = path    #input path until train/val
        self.img_path = path +'/Image/'
        self.label_path = path + '/Label/'

        self.img_list = [self.img_path+k for k in os.listdir(self.img_path)]
        self.label_list = [self.label_path + k for k in os.listdir(self.label_path)]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label_path = self.label_list[idx]
        img, label = Image.open(img_path), Image.open(label_path)
        
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
        
        return img, label

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),]
    )
    dataset = CustomDataset('C:/Users/ys499/Desktop/DL_implement/UNET/data/train', transform=transform)
    dataloader = d.DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=False)
    sample = []
    for epoch in range(2):
        print(f"epoch : {epoch} ")
        for batch in dataloader:
            img, label = batch
            sample.append(img)

    # plt.figure()
    figure, axis = plt.subplots(2,2)
    axis[0][0].imshow(np.transpose(sample[0][0],(1,2,0)), cmap = 'gray')
    axis[0][1].imshow(np.transpose(sample[1][0],(1,2,0)), cmap = 'gray')
    axis[1][0].imshow(np.transpose(sample[2][0],(1,2,0)), cmap = 'gray')
    axis[1][1].imshow(np.transpose(sample[3][0],(1,2,0)), cmap = 'gray')
    plt.show()
    
