import os
from socket import AF_IPX
import numpy as np
from PIL import Image, ImageSequence
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.ndimage as mor
from skimage.measure import label
import cv2
np.set_printoptions(threshold=np.inf)

def calWmap(mask):
    """
        Calculated pixelwise weights
        For detecting cell membranes
        Assume that d2 is very closed to cells on which have d1 distance at given pixels
    """
    img_size = mask.shape[0]*mask.shape[1]
    
    # Cal w_c
    n_pixel = np.count_nonzero(mask)    # membrane number
    img_scalered = (mask/n_pixel) + (1-mask)/(img_size - n_pixel)
    w_c = img_scalered/max((1/n_pixel),(1/(img_size-n_pixel)))
    
    # Cal Disctance_map
    cells = label(1-mask, connectivity=1) 
    n_cluster = np.amax(cells)

    #d1, d2
    d1 = np.ones((mask.shape[0], mask.shape[1]))
    d2 = np.ones((mask.shape[0], mask.shape[1]))
    if n_cluster >=2:
        d1[ :, :] = mor.distance_transform_edt((1-cells)[:,:]== 1)
        d2 = np.sqrt((np.multiply(d1,d1) + d2))

    return w_c + 10*np.exp(-(np.multiply((d1+d2),(d1+d2))/50))


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
    """
        If you don't use Overlap-tile strategy,
        Just load images, and save it
        i.e., Below code have to be changed
    """
    train_img, label_img = loadImage()
    label_scaled = label_img[:,0:572, 0:572]
    


    # Save Image 
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

    # Save Label & weights
    for idx, picture in enumerate(label_scaled):
        result   = slice(overlapTile(picture))
        w_result = [calWmap(result[i]) for i in range(4)]
        os.makedirs(os.path.join(os.getcwd(),'UNET\\data\\train\\label\\'), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(),'UNET\\data\\train\\weights\\'), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(),'UNET\\data\\val\\label\\'), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(),'UNET\\data\\val\\weights\\'), exist_ok=True)

        for i in range(4):
            im = Image.fromarray(result[i])
            imw = Image.fromarray(w_result[i].astype(np.uint8))

            if idx < len(label_img)-3:
                title = os.path.join(os.getcwd(),'UNET\\data\\train\\label\\')+str(4*idx+i)+'.jpg'
                w_title= os.path.join(os.getcwd(),'UNET\\data\\train\\weights\\')+str(4*idx+i)+'.jpg'
            else:
                title = os.path.join(os.getcwd(),'UNET\\data\\val\\label\\')+str(4*(idx+3-len(label_img))+i)+'.jpg' 
                w_title = os.path.join(os.getcwd(),'UNET\\data\\val\\weights\\')+str(4*(idx+3-len(label_img))+i)+'.jpg' 
            im.save(title, 'JPEG')
            imw.save(w_title, 'JPEG')


