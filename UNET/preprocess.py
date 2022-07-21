import os
from socket import AF_IPX
import numpy as np
from PIL import Image, ImageSequence
from pathlib import Path
import matplotlib.pyplot as plt

def loadImage():
    """
        Load Images:
        Convert tif image file to individual image files ndarray list
    """
    dir_data = os.path.join(os.getcwd(),'UNET\\predata')
    origin_train_img = 'train-volume.tif'
    origin_label_img = 'train-labels.tif'

    train_imgs = Image.open(os.path.join(dir_data, origin_train_img))
    label_imgs = Image.open(os.path.join(dir_data, origin_label_img))
    train_img_list, label_img_list =[], []
    for _, page_image in enumerate(ImageSequence.Iterator(train_imgs)):
        train_img_list.append(np.array(page_image))
    for _, page_image in enumerate(ImageSequence.Iterator(label_imgs)):
        label_img_list.append(np.array(page_image))

    return np.stack(train_img_list), np.stack(label_img_list)

def overlapTile(img_arr):
    """ 
        Overlap-Tile Strategy:
        Implemented at Preprocessing stage
        for using it directly at train step
    """

    height, width = img_arr.shape[0], img_arr.shape[1]

    # Extend width
    lw_result = np.flip(img_arr[0:height, 0:92], axis = 1)
    rw_result = np.flip(img_arr[0:height, width-92:width], axis = 1)
    w_result = np.concatenate((np.concatenate((lw_result, img_arr), axis=1), rw_result), axis=1)

    # Extend height
    top_result = np.flip(w_result[0:92,:], axis = 0)
    bottom_result = np.flip(w_result[height-92:height, :], axis = 0)

    # Final result
    result = np.concatenate((np.concatenate((top_result,w_result),axis=0),bottom_result),axis=0)

    return result

def slice(img):
    """
        Slicing(pad):
        Crop image have size 572 * 572 (blue rectangle in paper)
    """

    img_list = [img[0:572, 0:572],
                img[0:572, 124:696],
                img[124:696, 0:572],
                img[124:696, 124:696]]
    return img_list


if __name__ == '__main__':
    train_img, label_img = loadImage()
    
    """
        If you don't use Overlap-tile strategy,
        Just load images, and save it
        i.e., Below code have to be changed
    """

    # Save train image 
    for idx, picture in enumerate(train_img):
        result = slice(overlapTile(picture))        
        os.makedirs(os.path.join(os.getcwd(),'UNET\\data\\train\\Image\\'), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(),'UNET\\data\\val\\Image\\'), exist_ok=True)

        for i in range(4):
            im = Image.fromarray(result[i])
            if idx < len(train_img)-3:
                title = os.path.join(os.getcwd(),'UNET\\data\\train\\Image\\')+str(4*idx+i)+'.jpg'
            else:
                title = os.path.join(os.getcwd(),'UNET\\data\\val\\Image\\')+str(4*(idx+3-len(train_img))+i)+'.jpg'
            im.save(title, 'JPEG')

    # Save validation image
    for idx, picture in enumerate(label_img):
        result = slice(overlapTile(picture))
        os.makedirs(os.path.join(os.getcwd(),'UNET\\data\\train\\label\\'), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(),'UNET\\data\\val\\label\\'), exist_ok=True)

        for i in range(4):
            im = Image.fromarray(result[i])
            if idx < len(label_img)-3:
                title = os.path.join(os.getcwd(),'UNET\\data\\train\\label\\')+str(4*idx+i)+'.jpg'
            else:
                title = os.path.join(os.getcwd(),'UNET\\data\\val\\label\\')+str(4*(idx+3-len(label_img))+i)+'.jpg' 
            im.save(title, 'JPEG')


