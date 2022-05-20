import torch
import torch.nn as nn
import torch.utils.data
from alexnet import *
import data
import numpy as np

import matplotlib.pyplot as plt

def set_seed():
    manualseed = 999
    random.seed(manualseed)
    torch.manual_seed(manualseed)

def softmax(x):
    x = x.cpu()
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x
if __name__ == '__main__':
    set_seed()
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    model_dir = 'C:/Users/ys499/Desktop/DL_implement/ALEXNET/Exp/2205201510/87th_best_model.ptl'
    data_dir = 'C:/Users/ys499/data'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    test_loader = data.make_test_datasets(batchsize= 128, dataroot= data_dir)

    model = Alexnet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    epoch = 0
    correct = 0
    samples = 0
    test_loss = 0
    model.load_state_dict(torch.load(model_dir)['state_dict']) 

    model.eval()

    with torch.no_grad():
        for i, (datas, target) in enumerate(test_loader,0):
            datas, target = datas.to(device), target.to(device)
            output = model(datas)
            test_loss += criterion(output, target).item()
            _, predict = torch.max(output.data, 1)
            correct += (predict == target).sum().item()
            samples += predict.size(0)
            if (i == 0 or i == len(test_loader)-1):
                img = datas[0].cpu().numpy().T
                data_min = np.min(img, axis=(1,2), keepdims=True)
                data_max = np.max(img, axis=(1,2), keepdims=True)
                scaled_data = (img - data_min) / (data_max - data_min)
                output = softmax(output[0])
                print(output)
                plt.imshow(scaled_data)
                plt.show()

        test_loss /= samples
        print("\nTest set: Average_Loss: {:.4f},Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, samples, 100. * correct / samples))
    
        print('='*50)