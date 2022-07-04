import enum
from pickletools import optimize
from random import sample
import resnet
import params as params
import resnet
import data

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def main():
    train_loader = data.make_train_datasets(256, params.train_path)
    val_loader = data.make_val_datasets(256, params.valid_path)
    writer = SummaryWriter(params.save_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = resnet.ResNet(resnet.ResBlock).to(device)
    model = nn.DataParallel(model, output_device=0)


    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    preval_error = 0
    for epoch in range(params.max_epoch):
        print(f"Epoch: {epoch}")
        
        # train
        model.train()
        train_loss = 0
        pbar1 = tqdm(enumerate(train_loader))
        for i, (image, label) in pbar1:
            optimizer.zero_grad()
            image , label = image.to(device), label.to(device)
            feature = model(image)
            loss = criterion(feature, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar1.set_postfix({'loss': loss})
        train_loss /= i
        
        correct = 0
        samples = 0
        # test
        with torch.no_grad():
            model.eval()
            val_loss = 0
            pbar2 = tqdm(enumerate(val_loader))
            for i, (image, label) in pbar2:
                image , label = image.to(device), label.to(device)
                feature = model(image)
                loss = criterion(feature, label)
                val_loss += loss.item()
                _, predict = torch.max(feature.data, 1)
                correct += (predict == label).sum().item()
                samples += predict.size(0)      

                pbar2.set_postfix({'loss': loss})
            val_loss /= i
        currval_error = 100.*correct/samples

        if abs(currval_error - preval_error)<0.01:
            optimizer.param_groups[0]['lr']/=10
            print(optimizer.param_groups[0]['lr'])

        preval_error = currval_error

        print(f"Train Loss: {train_loss}\nValidation Loss: {val_loss}")
        print(f"Accuracy: {currval_error}")
            
        #TODO: train tensorboard (train_loss)
        writer.add_scalar('train_loss', train_loss, epoch)

        #TODO: valid tensorboard (val_loss)
        writer.add_scalar('valid_loss', val_loss, epoch)

        #TODO: accuracy tensorboard (currval_error)
        writer.add_scalar('accuracy (%)', currval_error, epoch)

        #TODO: epoch 10마다 저장 (epoch+1 % 10 == 0), 경로: model/VGG_{epoch+1}.pth
        if (((epoch+1) % 10 )== 0 or (epoch == 0)):
            os.makedirs(params.save_path +'model/', exist_ok= True)
            torch.save(model.state_dict(), params.save_path +'model/'+ (str)(epoch+1)+'.pth')

    writer.close()


if __name__ == "__main__":
    print("Main")
    main()

