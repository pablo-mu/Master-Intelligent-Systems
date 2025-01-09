import scipy.signal.windows
from PIL import Image
from scipy.ndimage import filters
from scipy.signal import medfilt2d
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import time
def gaussianFilter(im, sigma=5):
    # im is PIL image
    return filters.gaussian_filter(im, sigma)

def manualgaussianFilter(im, sigma = 5):
    gv1d = scipy.signal.windows.gaussian(im.shape[0], sigma).reshape(im.shape[0],1)
    gv2d = gv1d * np.transpose(gv1d)
    mask = gv2d / np.sum(gv2d)
    return filters.convolve(im, mask)

def manualsepgaussianFilter(im, sigma=5):
    gv1d = scipy.signal.windows.gaussian(im.shape[0], sigma).reshape(im.shape[0],1)
    gv1d = gv1d / np.sum(gv1d)
    filtered_rows = filters.convolve(im, gv1d)
    return filters.convolve(filtered_rows, np.transpose(gv1d)/np.sum(gv1d))

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


if __name__ == "__main__":
    normal_time = []
    manual_time = []
    manual_sep_time = []
    im_pil = Image.open('./imgs-P2/lena256.pgm').convert('L')
    im = np.array(im_pil)
    im = addGaussianNoise(im, sd=5)
    for sigma in [1,3,5,7]:
        st = time.time()
        gaussianFilter(im,sigma)
        et = time.time()
        normal_time.append(et-st)
        st = time.time()
        im2 = manualgaussianFilter(im,sigma)
        Image.fromarray(im2).show()
        et = time.time()
        manual_time.append(et-st)
        st = time.time()
        im3 = manualsepgaussianFilter(im, sigma)
        Image.fromarray(im3).show()
        et = time.time()
        manual_sep_time.append(et - st)

    a=plt.plot([1,3,5,7],normal_time, label = 'Normal Time')
    b=plt.plot([1,3,5,7], manual_time, label ='Sep Time')
    c = plt.plot([1,3,5,7], manual_sep_time, label ='Sep Time')
    plt.ylabel('Time of execution')
    plt.xlabel('Filter Size')
    plt.legend(['Normal Time','Manual Time','Manual Sep Time'])
    plt.show()

