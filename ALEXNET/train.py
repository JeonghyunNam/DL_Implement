from multiprocessing import reduction
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import random
import os
import datetime
import matplotlib.pyplot as plt

from alexnet import Alexnet
from data import *

def set_seed():
    manualseed = 999
    random.seed(manualseed)
    torch.manual_seed(manualseed)

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.to(device)
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader, 0):
        target = target.type(torch.LongTensor)

        # img = data[0].numpy().T
        # data_min = np.min(img, axis=(1,2), keepdims=True)
        # data_max = np.max(img, axis=(1,2), keepdims=True)
        # scaled_data = (img - data_min) / (data_max - data_min)
        # plt.imshow(scaled_data)
        # plt.show()
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, criterion, save_root, epoch, best_accuracy):
    model.eval()
    correct = 0
    samples = 0
    test_loss = 0

    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader,0):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predict = torch.max(output.data, 1)
            correct += (predict == target).sum().item()
            samples += predict.size(0)

        test_loss /= samples
        print("\nTest set: Average_Loss: {:.4f},Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, samples, 100. * correct / samples))
        
        if best_accuracy < 100 * correct / samples :
            best_accuracy = 100 * correct / samples
            print("\nBest accuracy updated to model.ptl")
            torch.save({
            "epoch": epoch,
            "loss": test_loss,
            "accuracy": best_accuracy,
            "state_dict": model.state_dict()
            }, save_root+str(epoch)+'th_best_model.ptl')
    
        print('='*50)
        return best_accuracy


if __name__ == '__main__':
    set_seed()
    date = datetime.datetime.now()
    data_root = 'C:/Users/ys499/data'
    result_root = 'C:/Users/ys499/Desktop/DL_implement/ALEXNET/Exp/'
    model_save_root = result_root + str(date.strftime('%y%m%d%H%M'))+'/'
    os.makedirs(model_save_root, exist_ok= True)

    num_classes = 10
    num_epoches = 90
    batch_size = 128
    momentum = 0.9
    weight_decay = 0.0005
    learning_rate = 0.0001  # heuristic learning rate

    traindataloader = make_train_datasets(batch_size, data_root)    # Train data
    testdataloader  = make_test_datasets(batch_size, data_root)     # Test data

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # optimizer = optim.SGD(alexnet.parameters(), lr = learning_rate, momentum= momentum, weight_decay= weight_decay)
    # SGD does not work
    alexnet = Alexnet().to(device)
    criterion =torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(alexnet.parameters(), lr= learning_rate, betas = (0.5, 0.999))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 30, gamma= 0.1)

    best_accuracy = 0
    for epoch in range(1, num_epoches + 1):
        if epoch % 30 == 0:
            lr_scheduler.step()
        train(alexnet, device, traindataloader, criterion, optimizer, epoch)
        best_accuracy = test(alexnet, device, testdataloader, criterion, model_save_root, epoch, best_accuracy)