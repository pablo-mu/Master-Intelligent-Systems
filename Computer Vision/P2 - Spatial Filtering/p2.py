#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.signal.windows
from PIL import Image
from scipy.ndimage import filters
from scipy.signal import medfilt2d
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import os
import sys
import time

# sys.path.append("../../p1/code") # set the path for visualPercepUtils.py
import visualPercepUtils as vpu


# -----------------------
# Salt & pepper noise
# -----------------------

def addSPNoise(im, percent):
    # Now, im is a PIL image (not a NumPy array)
    # percent is in range 0-100 (%)

    # convert image it to numpy 2D array and flatten it
    im_np = np.array(im)
    im_shape = im_np.shape  # keep shape for later use (*)
    im_vec = im_np.flatten()  # this is a 1D array # https://www.geeksforgeeks.org/differences-flatten-ravel-numpy/

    # generate random locations
    N = im_vec.shape[0]  # number of pixels
    m = int(math.floor(percent * N / 100.0)) # number of pixels corresponding to the given percentage
    locs = np.random.randint(0, N, m)  # generate m random positions in the 1D array (index 0 to N-1)

    # generate m random S/P values (salt and pepper in the same proportion)
    s_or_p = np.random.randint(0, 2, m)  # 2 random values (0=salt and 1=pepper)

    # set the S/P values in the random locations
    im_vec[locs] = 255 * s_or_p  # values after the multiplication will be either 0 or 255

    # turn the 1D array into the original 2D image
    im2 = im_vec.reshape(im_shape) # (*) here is where we use the shape that we saved earlier

    # convert Numpy array im2 back to a PIL Image and return it
    return Image.fromarray(im2)


def testSandPNoise(im, percents):
    imgs = []
    for percent in percents:
        imgs.append(addSPNoise(im, percent))
    return imgs


# -----------------
# Gaussian noise
# -----------------

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

def testGaussianNoise(im, sigmas):
    imgs = []
    for sigma in sigmas:
        print('testing sigma:', sigma)
        imgs.append(addGaussianNoise(im, sigma))
        print(len(imgs))
    return imgs


# -------------------------
# Average (or mean) filter
# -------------------------

def averageFilter(im, filterSize):
    mask = np.ones((filterSize, filterSize))
    mask = np.divide(mask, np.sum(mask)) # can you think of any alternative for np.sum(mask)?
    return filters.convolve(im, mask)

def averageFilterSep(im, filterSize):
    row_mask = (1/filterSize)*np.ones((filterSize, 1))
    filtered_rows = filters.convolve(im, row_mask)
    col_mask = ((1 / filterSize) * np.ones((1, filterSize)))
    return filters.convolve(filtered_rows, col_mask)



def testAverageFilter(im_clean, params):
    imgs = []
    for sp_pctg in params['sp_pctg']:
        im_dirty = addSPNoise(im_clean, sp_pctg) # salt and pepper noise
        for filterSize in params['filterSizes']:
            imgs.append(np.array(im_dirty))
            imgs.append(averageFilter(im_dirty, filterSize))
    return imgs


# -----------------
# Gaussian filter
# -----------------
def gaussianFilter(im, sigma=5):
    # im is PIL image
    return filters.gaussian_filter(im, sigma)

def manualgaussianFilter(im, sigma = 5, n=15):
    gv1d = scipy.signal.windows.gaussian(n, sigma).reshape(1, -1)
    gv2d = gv1d * np.transpose(gv1d)
    return filters.convolve(im, gv2d)
    # plt.imshow(gv1d, interpolation='none', cmap=plt.colormaps['jet'],aspect='equal')
    # plt.title('1D Gaussian')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    # plt.imshow(gv2d, interpolation='none', cmap=plt.colormaps['jet'],aspect='equal')
    # plt.title('2D Gaussian')
    # plt.show()


def manualsepgaussianFilter(im, sigma=5):
    gv1d = scipy.signal.windows.gaussian(im.shape[0], sigma).reshape(im.shape[0],1)
    filtered_rows = filters.convolve(im, gv1d)
    return filters.convolve(filtered_rows, np.transpose(gv1d))


    # plt.imshow(image,interpolation='None')
    # plt.plot(x, gv1d, 'bo-', markersize=5, linestyle='--', linewidth=1, color='blue')
    # plt.title('Gaussian Window')
    # plt.xlabel('Index')
    # plt.ylabel('Value')


def testGaussianFilter(im_clean, params):
    # This function turned out to be too similar to testAverageFilter
    # This is a good sign that code factorization is called for :)
    imgs = []
    for sigma in params['sd_gauss_noise']:
        im_dirty = addGaussianNoise(im_clean, sigma)
        for filterSize in params['sd_gauss_filter']:
            imgs.append(np.array(im_dirty))
            imgs.append(gaussianFilter(im_dirty, filterSize))
    return imgs

# -----------------
# Median filter
# -----------------

def medianFilter(im, filterSize):
    return medfilt2d(im, filterSize)

def testMedianFilter(im_clean, params):
    # This function turned out to be too similar to testAverageFilter
    # This is a good sign that code factorization is called for :)
    imgs = []
    for sp_pctg in params['sp_pctg']:
        im_dirty = addSPNoise(im_clean, sp_pctg)
        for filterSize in params['filterSizes']:
            imgs.append(np.array(im_dirty))
            imgs.append(medianFilter(im_dirty, filterSize))
    return imgs

def quotientImage(im, sigma):
    filtered_im = filters.gaussian_filter(im, sigma)
    return np.divide(im, filtered_im)

def testquotientImage(im_clean,params):
    imgs = []
    for sp_pctg in [0,0,0]:
        im_dirty = addSPNoise(im_clean, sp_pctg)
        for filterSize in params['sd_gauss_filter']:
            imgs.append(np.array(im_dirty))
            imgs.append(quotientImage(im_dirty, filterSize))
            Image.fromarray(quotientImage(im_dirty, filterSize)).show()
    return imgs

# -----------------
# Test image files
# -----------------

path_input = './imgs-P2/'
path_output = './imgs-out-P2/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.pgm")
else:
    files = [path_input + 'lena256.pgm']  # lena256, lena512

# --------------------
# Tests to perform
# --------------------

testsNoises = ['testSandPNoise', 'testGaussianNoise']
testsFilters = ['testAverageFilter', 'testGaussianFilter', 'testMedianFilter', 'testquotientImage']
bAllTests = True
if bAllTests:
    tests = testsNoises + testsFilters
else:
    tests = ['testSandPNoise']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testGaussianNoise': 'Gaussian noise',
             'testSandPNoise': 'Salt & Pepper noise',
             'testAverageFilter': 'Mean filter',
             'testGaussianFilter': 'Gaussian filter',
             'testMedianFilter': 'Median filter',
             'testquotientImage': 'Quotient Image'}

bSaveResultImgs = False

# -----------------------
# Parameters of noises
# -----------------------
percentagesSandP = [3]  # ratio (%) of image pixes affected by salt and pepper noise
gauss_sigmas_noise = [3, 5, 10]  # standard deviation (for the [0,255] range) for Gaussian noise

# -----------------------
# Parameters of filters
# -----------------------

gauss_sigmas_filter = [1.2]  # standard deviation for Gaussian filter
avgFilter_sizes = [3, 7, 15]  # sizes of mean (average) filter
medianFilter_sizes = [3, 7, 15]  # sizes of median filter

testsUsingPIL = ['testSandPNoise']  # which test(s) uses PIL images as input (instead of NumPy 2D arrays)


# -----------------------------------------
# Apply defined tests and display results
# -----------------------------------------

def doTests():
    print("Testing on", files)
    for imfile in files:
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array

        for test in tests:

            if test == "testGaussianNoise":
                params = gauss_sigmas_noise
                subTitle = r", $\sigma$: " + str(params)
            elif test == "testSandPNoise":
                params = percentagesSandP
                subTitle = ", %: " + str(params)
            elif test == "testAverageFilter":
                params = {}
                params['filterSizes'] = avgFilter_sizes
                params['sp_pctg'] = percentagesSandP
                subTitle = ", " + str(params)
            elif test == "testMedianFilter":
                params = {}
                params['filterSizes'] = avgFilter_sizes
                params['sp_pctg'] = percentagesSandP
                subTitle = ", " + str(params)
            elif test == "testGaussianFilter":
                params = {}
                params['sd_gauss_noise'] = [0,0,0]
                params['sd_gauss_filter'] = gauss_sigmas_filter
                subTitle = r", $\sigma_n$ (noise): " + str(gauss_sigmas_noise) + ", $\sigma_f$ (filter): " + str(gauss_sigmas_filter)
            elif test == "testquotientImage":
                params = {}
                params['sd_gauss_noise'] = gauss_sigmas_noise
                params['sd_gauss_filter'] = gauss_sigmas_filter
                subTitle = r", $\sigma_n$ (noise): " + str(gauss_sigmas_noise) + ", $\sigma_f$ (filter): " + str(
                    gauss_sigmas_filter)

            if test in testsUsingPIL:
                outs_pil = eval(test)(im_pil, params)
                outs_np = vpu.pil2np(outs_pil)
            else:
                # apply test to given image and given parameters
                outs_np = eval(test)(im, params)
                print("num images", len(outs_np))
            print(len(outs_np))
            # display original image, noisy images and filtered images
            vpu.showInGrid([im] + outs_np, title=nameTests[test] + subTitle)


if __name__ == "__main__":
    doTests()









    # normal_time = []
    # sep_time = []
    # filter_sizes = [1,3,5,7,9,11,13,15,25]
    # im_pil = Image.open('./imgs-P2/lena512.pgm').convert('L')
    # im = np.array(im_pil)
    # im = addGaussianNoise(im, sd=5)
    # for size in filter_sizes:
    #     st = time.time()
    #     im2 = averageFilter(im,size)
    #     et = time.time()
    #     normal_time.append(et-st)
    #     st = time.time()
    #     im3 = averageFilterSep(im,size)
    #     et = time.time()
    #     vpu.showInGrid([im2,im3])
    #     sep_time.append(et-st)
    # print(normal_time)
    # print(sep_time)
    # a=plt.plot(filter_sizes,normal_time, label = 'Normal Time')
    # b=plt.plot(filter_sizes, sep_time, label ='Sep Time')
    # plt.ylabel('Time of execution')
    # plt.xlabel('Filter Size')
    # plt.legend(['Normal Time','Sep Time'])
    # plt.show()
