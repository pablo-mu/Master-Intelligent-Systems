import scipy.signal.windows
from PIL import Image
from scipy.ndimage import filters
from scipy.signal import medfilt2d
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob

def addGaussianNoise(im, sd=5):
    im2 = Image.fromarray(im)
    if im2.mode == 'L':
        return im + np.random.normal(loc=0, scale=sd, size=im.shape)
    else:
        im2 = im2.convert('RGB')
        im3 = np.array(im2)
        im3[:, :, 0] = im[:, :, 0] + np.random.normal(loc=0, scale=sd, size=(im.shape[0], im.shape[1]))
        im3[:, :, 1] = im[:, :, 1]+ np.random.normal(loc=0, scale=sd, size=(im.shape[0], im.shape[1]))
        im3[:, :, 2] = im[:, :, 2] + np.random.normal(loc=0, scale=sd, size=(im.shape[0], im.shape[1]))
        return im3

def gaussianFilter(im, sigma=5):
    # im is PIL image
    return filters.gaussian_filter(im, sigma)


if __name__ == "__main__":
    im = np.array(Image.open('./imgs-P2/foto3.jpg'))
    im2 = addGaussianNoise(im,5)
    filtered_red = np.copy(im2)
    filtered_red[:,:,0] = gaussianFilter(im2[:,:,0],5)
    filtered_green = np.copy(im2)
    filtered_green[:, :, 1] = gaussianFilter(im2[:, :, 1], 5)
    filtered_blue = np.copy(im2)
    filtered_blue[:, :, 2] = gaussianFilter(im2[:, :, 2], 5)
    filtered_im = np.copy(im2)
    filtered_im[:,:,0] = filtered_red[:,:,0]
    filtered_im[:,:,1] = filtered_green[:,:,1]
    filtered_im[:,:,2] = filtered_blue[:,:,2]
    Image.fromarray(im2,mode='RGB').save('./imgs-P2/fotonoise.jpg')
    Image.fromarray(filtered_red,mode='RGB').save('./imgs-P2/foto3filterred.jpg')
    Image.fromarray(filtered_blue,mode='RGB').save('./imgs-P2/foto3filterblue.jpg')
    Image.fromarray(filtered_green,mode='RGB').save('./imgs-P2/foto3filtergreen.jpg')
    Image.fromarray(filtered_im,mode='RGB').save('./imgs-P2/foto3filter.jpg')
