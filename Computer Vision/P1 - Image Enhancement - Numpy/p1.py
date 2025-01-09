#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import glob
import os
import visualPercepUtils as vpu
import warnings


def histeq(im, nbins=256):
    imhist, bins = np.histogram(im.flatten(), list(range(nbins)), density=False)
    cdf = imhist.cumsum() # cumulative distribution function (CDF) = cummulative histogram
    factor = 255 / cdf[-1]  # cdf[-1] = last element of the cummulative sum = total number of pixels)
    im2 = np.interp(im.flatten(), bins[:-1], factor*cdf)
    return im2.reshape(im.shape), cdf

def testHistEq(im):
    im2, cdf = histeq(im)
    return [im2, cdf]

def darkenImg(im,p=2):
    im2 = Image.fromarray(im)
    if im2.mode == 'L':
        return (im ** float(p)) / (255 ** (p - 1)) # try without the float conversion and see what happens
    else:
        im2 = im2.convert('RGB')
        im3 = np.array(im2)
        im3[:,:,0] = (im3[:,:,0] ** float(p)) / (255 ** (p - 1))
        im3[:,:,1] = (im3[:,:,1] ** float(p)) / (255 ** (p - 1))
        im3[:,:,2] = (im3[:,:,2] ** float(p)) / (255 ** (p - 1))
        return im3
def brightenImg(im,p=2):
    im2 = Image.fromarray(im)
    if im2.mode == 'L':
        return np.power(255.0 ** (p - 1) * im, 1. / p)  # notice this NumPy function is different to the scalar math.pow(a,b)
    else:
        im2 = im2.convert('RGB')
        im3 = np.array(im2)
        im3[:, :, 0] = np.power(255.0 ** (p - 1) * im3[:, :, 0], 1. / p)
        im3[:, :, 1] = np.power(255.0 ** (p - 1) * im3[:, :, 1], 1. / p)
        im3[:, :, 2] = np.power(255.0 ** (p - 1) * im3[:, :, 2], 1. / p)
        return im3


def checkBoardImg(im, m, n):
    # Get the dimensions of the input image
    height, width = im.shape[:2]
    # Calculate the size of each cell
    cell_height = height // m
    cell_width = width // n
    # Initialize the output image
    output_img = np.copy(im)
    # Loop through the cells and invert pixels alternatively
    for i in range(0, height, cell_height):
        for j in range(0, width, cell_width):
            # Calculate the coordinates of the cell boundaries
            cell_top = i
            cell_bottom = min(i + cell_height, height)
            cell_left = j
            cell_right = min(j + cell_width, width)
            # Invert the pixels in the current cell if (i+j) is odd
            if (i // cell_height + j // cell_width) % 2 == 1:
                output_img[cell_top:cell_bottom, cell_left:cell_right] = 255 - im[cell_top:cell_bottom, cell_left:cell_right]
    return output_img
#def checkBoardImg(im, m, n):
#    im_shape = im.shape
#     output_img = np.copy(im)
#     row_block, col_block = np.linspace(0,im_shape[0],m+1,dtype = int), np.linspace(0,im_shape[1],n,dtype=int)
#     for i in range(m):
#         for j in range(n):
#             if i+j % 2 == 1: # Invert the pixels in the current cell if (i+j) is odd
#                 output_img[row_block[i] : row_block[i+1], col_block[j] : col_block[j+1]] = 255 - im[row_block[i]:row_block[i+1], col_block[j]:col_block[j+1]]

def multiHist(im, n):
    list_hist = []
    im_shape = im.shape
    for i in range(1,n+1):
        row_block, col_block = np.linspace(0, im_shape[0], 2 ** (i - 1)+1, dtype = int), np.linspace(0, im_shape[1], 2 ** (i - 1) +1, dtype = int)
        for j in range(2 ** (i - 1)):
            for k in range(2 ** (i - 1)):
                hist, bins = np.histogram(im[row_block[j]: row_block[j+1], col_block[k]: col_block[k+1]].flatten(), bins=3, density=False)
                list_hist.append(hist)
    return list_hist

def expTransf(alpha, n, l0,l1,bInc = True):
    # Set 'a' and 'b' to guarantee the range for bInc = True
    warnings.filterwarnings('ignore')
    a = (l1 - l0) * np.exp(alpha * l0 ** 2)
    b = l0
    l_values = np.linspace(l0, l1, n, dtype=int)
    if bInc:
        T_values = a * np.exp(-alpha * (l_values ** 2)) + b
    else:
        T_values = T_values[::-1]
    return T_values
#visualitzar-lo pa vore si está bé

#
def transfImage(im,f):
    values = f
    imhist, bins = np.histogram(im.flatten(), list(range(len(values)+1)), density=False)
    im2 = np.interp(im.flatten(), bins[:-1], values)
    return im2.reshape(im.shape)





def testCheckBoard(im):
    m,n= 5,6
    im2 = checkBoardImg(im,m,n)
    return [im2]

def testDarkenImg(im):
    im2 = darkenImg(im,p=2) #  Is "p=2" different here than in the function definition? Can we remove "p=" here?
    return [im2]


def testBrightenImg(im):
    p=2
    im2=brightenImg(im,p)
    return [im2]

def saveImg(imfile, test, im2):
    dirname, basename = os.path.dirname(imfile), os.path.basename(imfile)
    fname, fext = os.path.splitext(basename)
    # print(dname,basename)
    pil_im = Image.fromarray(im2.astype(np.uint8))  # from array to Image
    pil_im.save(path_output + '//' + fname + suffixFiles[test] + fext)

path_input = './imgs-P1/'
path_output = './imgs-out-P1/'
bAllFiles = True
if bAllFiles:
    files = glob.glob(path_input + "*.pgm")
else:
    files = [path_input + 'iglesia.pgm'] # iglesia,huesos

bAllTests = True
if bAllTests:
    tests = ['testHistEq', 'testBrightenImg', 'testDarkenImg','testCheckBoard']
else:
    tests = ['testHistEq']#['testBrightenImg']
nameTests = {'testHistEq': "Histogram equalization",
             'testBrightenImg': 'Brighten image',
             'testDarkenImg': 'Darken image',
             'testCheckBoard': 'CheckBoard Image'
             }
suffixFiles = {'testHistEq': '_heq',
               'testBrightenImg': '_br',
               'testDarkenImg': '_dk',
               'testCheckBoard': '_cb'}

bSaveResultImgs = True

def doTests():
    print("Testing on", files)
    for imfile in files:
        #im = np.array(Image.open(imfile).convert('L'))  # from Image to array
        im = np.array(Image.open(imfile))
        for test in tests:
            out = eval(test)(im)
            im2 = out[0]
            vpu.showImgsPlusHists(im, im2, title=nameTests[test])
            if len(out) > 1:
                vpu.showPlusInfo(out[1],"cumulative histogram" if test=="testHistEq" else None)
            saveImg(imfile, test, im2)


if __name__== "__main__":
    doTests()

