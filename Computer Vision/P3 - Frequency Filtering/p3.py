#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import scipy.signal.windows
from scipy.ndimage import filters
import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import os
import sys


#sys.path.append("../../p1/code") set the path for visualPercepUtils.py
import visualPercepUtils as vpu

# ----------------------
# Fourier Transform
# ----------------------


def FT(im):
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
    return fft.fftshift(fft.fft2(im))  # perform also the shift to have lower frequencies at the center


def IFT(ft):
    return fft.ifft2(fft.ifftshift(ft))  # assumes ft is shifted and therefore reverses the shift before IFT


def testFT(im, params=None):
    ft = FT(im)
    #print(ft.shape)
    phase = np.angle(ft)
    magnitude = np.log(np.absolute(ft))
    bMagnitude = True
    if bMagnitude:
        im2 = np.absolute(IFT(ft))  # IFT consists of complex number. When applied to real-valued data the imaginary part should be zero, but not exactly for numerical precision issues
    else:
        im2 = np.real(IFT(ft)) # with just the module we can't appreciate the effect of a shift in the signal (e.g. if we use fftshift but not ifftshift, or viceversa)
        # Important: one case where np.real() is appropriate but np.absolute() is not is where the sign in the output is relevant
    return [magnitude, phase, im2]


# -----------------------
# Convolution theorem
# -----------------------

# the mask corresponding to the average (mean) filter
def avgFilter(filterSize):
    mask = np.ones((filterSize, filterSize))
    return mask/np.sum(mask)


# apply average filter in the spatial domain
def averageFilterSpace(im, filterSize):
    return filters.convolve(im, avgFilter(filterSize))


# apply average filter in the frequency domain
def averageFilterFrequency(im, filterSize):
    filterMask = avgFilter(filterSize)  # the usually small mask
    filterBig = np.zeros_like(im, dtype=float)  # as large as the image (dtype is important here!)

    # Now, place filter (the "small" filter mask) at the center of the "big" filter

    ## First, get sizes
    w, h = filterMask.shape
    w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2  # half width and height of the "big" mask

    ## Then, paste the small mask at the center using the sizes computed before as an aid
    filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterMask

    # FFT of the big filter
    filterBig = fft.ifftshift(filterBig)  # shift origin at upper-left corner

    # Finally, IFT of the element-wise product of the FT's
    return np.absolute(IFT(FT(im) * FT(filterBig)))  # both '*' and multiply() perform elementwise product


def testConvTheo(im, params=None):
    filterSize = params['filterSize']

    # image filtered with a convolution in spatial domain
    imFiltSpace = averageFilterSpace(im, filterSize)

    # image filtered in frequency domain
    imFiltFreq = averageFilterFrequency(im, filterSize)

    # How much do they differ?
    # To quantify the difference, we use the Root Mean Square Measure (https://en.wikipedia.org/wiki/Root_mean_square)
    margin = 5  # exclude some outer pixels to reduce the influence of border effects
    rms = np.linalg.norm(imFiltSpace[margin:-margin, margin:-margin] - imFiltFreq[margin:-margin, margin:-margin], 2) / np.prod(im.shape)
    print("Images filtered in space and frequency differ in (RMS):", rms)
    return [imFiltSpace, imFiltFreq]

## fer-lo com en l'anterior. Ficar-lo en una imatge de la grandària de la imatge
# i ficar-lo en el centre i ahi ja aplicar transformada i tot aixó.

def testgaussianFT(im, params=None):
     filterSize = params['filterSize']
     filterMask = gaussianFilter(filterSize,5)  # the usually small mask
     filterBig = np.zeros_like(im, dtype=float)

     ## First, get sizes
     w, h = filterMask.shape
     w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
     W, H = filterBig.shape
     W2, H2 = W / 2, H / 2  # half width and height of the "big" mask

     ## Then, paste the small mask at the center using the sizes computed before as an aid
     filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterMask

     # plt.imshow(filterBig)
     # plt.title("FilterBig")
     # plt.show()

     # FFT of the big filter
     filterBig = FT(fft.ifftshift(filterBig))
     phase = np.angle(filterBig)
     magnitude = np.absolute(filterBig)
     im2 = FT(im)
     im2 = im2 * filterBig
     im_filt_phase = np.angle(im2)
     im_filt_magnitude = np.absolute(im2)
     return [magnitude, phase, im_filt_magnitude, im_filt_phase]

# -----------------------
# Convolution theorem Gaussian Filter
# ----------------------

def gaussianFilter(filterSize,sigma = 5):
    gv1d = scipy.signal.windows.gaussian(filterSize, sigma).reshape(filterSize, 1)
    gv2d = np.outer(gv1d, gv1d)
    return gv2d/np.sum(gv2d)

def gaussianFilterSpace(im, filterSize, sigma = 5):
    gf = gaussianFilter(filterSize, sigma)
    result = filters.convolve(im, gf)
    return result

def gaussianFilterFrequency(im, filterSize, sigma = 5):
    filterMask = gaussianFilter(filterSize, sigma)  # the usually small mask
    filterBig = np.zeros_like(im, dtype=float)  # as large as the image (dtype is important here!)

    # Now, place filter (the "small" filter mask) at the center of the "big" filter

    ## First, get sizes
    w, h = filterMask.shape
    w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2  # half width and height of the "big" mask

    ## Then, paste the small mask at the center using the sizes computed before as an aid
    filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterMask

    # FFT of the big filter
    filterBig = fft.ifftshift(filterBig)  # shift origin at upper-left corner

    # Finally, IFT of the element-wise product of the FT's
    return np.absolute(IFT(FT(im) * FT(filterBig)))  # both '*' and multiply() perform elementwise product


def testgaussianConvTheo(im, params=None, sigma = 5):
    filterSize = params['filterSize']

    # image filtered with a convolution in spatial domain
    imFiltSpace = gaussianFilterSpace(im, filterSize, sigma)
    Image.fromarray(imFiltSpace,mode ='L').show()
    # image filtered in frequency domain
    imFiltFreq = gaussianFilterFrequency(im, filterSize, sigma)

    # How much do they differ?
    # To quantify the difference, we use the Root Mean Square Measure (https://en.wikipedia.org/wiki/Root_mean_square)
    margin = 5  # exclude some outer pixels to reduce the influence of border effects
    rms = np.linalg.norm(imFiltSpace[margin:-margin, margin:-margin] - imFiltFreq[margin:-margin, margin:-margin], 2) / np.prod(im.shape)
    print("Images filtered in space and frequency differ in (RMS):", rms)

    return [imFiltSpace, imFiltFreq]

# -----------------------------------
# High-, low- and band-pass filters
# -----------------------------------

# generic band-pass filter (both, R and r, given) which includes the low-pass (r given, R not)
# and the high-pass (R given, r not) as particular cases
def bandPassFilter(shape, r=None, R=None):
    n, m = shape
    m2, n2 = np.floor(m / 2.0), np.floor(n / 2.0)
    [vx, vy] = np.meshgrid(np.arange(-m2, m2 + 1), np.arange(-n2, n2 + 1))
    distToCenter = np.sqrt(vx ** 2.0 + vy ** 2.0)
    if R is None:  # low-pass filter assumed
        assert r is not None, "at least one size for filter is expected"
        filter = distToCenter<r # same as np.less(distToCenter, r)
    elif r is None:  # high-pass filter assumed
        filter = distToCenter>R # same as np.greater(distToCenter, R)
    else:  # both, R and r given, then band-pass filter
        if r > R:
            r, R = R, r  # swap to ensure r < R (alternatively, warn the user, or throw an exception)
        filter = np.logical_and(distToCenter<R, distToCenter>r)
    filter = filter.astype('float')  # convert from boolean to float. Not strictly required

    bDisplay = True
    if bDisplay:
        plt.imshow(filter, cmap='gray')
        plt.show()
        plt.title("The filter in the frequency domain")
        # Image.fromarray((255*filter).astype(np.uint8)).save('filter.png')

    return filter


def testBandPassFilter(im, params=None):
    r, R = params['r'], params['R']
    filterFreq = bandPassFilter(im.shape, r, R)  # this filter is already in the frequency domain
    filterFreq = fft.ifftshift(filterFreq)  # shifting to have the origin as the FT(im) will be
    return [np.absolute(fft.ifft2(filterFreq * fft.fft2(im)))]  # the filtered image

def mymask(n):
    mask = np.tri(n,n,-1) - np.transpose(np.tri(n,n,-1))
    return mask

def my_filter(im,n):
    filterMask = mymask(n)  # the usually small mask
    filterBig = np.zeros_like(im, dtype=float)  # as large as the image (dtype is important here!)

    # Now, place filter (the "small" filter mask) at the center of the "big" filter

    ## First, get sizes
    w, h = filterMask.shape
    w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2  # half width and height of the "big" mask

    ## Then, paste the small mask at the center using the sizes computed before as an aid
    filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterMask

    # FFT of the big filter
    filterBig = fft.ifftshift(filterBig)  # shift origin at upper-left corner

    # Finally, IFT of the element-wise product of the FT's
    return np.absolute(IFT(FT(im) * FT(filterBig)))

def testmymask(im,params = None):
    n = params['filterSize']
    return [my_filter(im,n)]

# -----------------
# Test image files
# -----------------
path_input = './imgs-P3/'
path_output = './imgs-out-P3/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.pgm")
else:
    files = [path_input + 'einstein.jpg']  # lena255, habas, mimbre

# --------------------
# Tests to perform
# --------------------
bAllTests = True
if bAllTests:
    tests = ['testFT','testgaussianFT','testConvTheo','testgaussianConvTheo', 'testBandPassFilter','testmymask']
else:
    tests = ['testFT']
    tests = ['testgaussianFT']
    tests = ['testConvTheo']
    tests = ['testgaussianConvTheo']
    tests = ['testBandPassFilter']
    tests = ['testmymask']
# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testFT': '2D Fourier Transform',
             'testgaussianFT': 'Gaussian Filter Transform of Fourier',
             'testConvTheo': 'Convolution Theorem (tested on mean filter)',
             'testgaussianConvTheo': 'Convolution Theorem (tested on gaussian filter)',
             'testBandPassFilter':'Frequency-based filters ("high/low/band-pass")',
             'testmymask' : 'Test MyMask'
             }

bSaveResultImgs = False

testsUsingPIL = []  # which test(s) uses PIL images as input (instead of NumPy 2D arrays)


# -----------------------------------------
# Apply defined tests and display results
# -----------------------------------------

def doTests():
    print("Testing on", files)
    for imfile in files:
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array

        for test in tests:
            if test is "testFT":
                params = {}
                subTitle = ": I, |F|, ang(F), IFT(F)"

            elif test is "testgaussianFT":
                params = {}
                params['filterSize'] = 7
                subTitle = ": I, |FT(M)|, ang(FT(M)), |FT(I) * FT(M)|, ang(FT(I)*FT(M))"

            elif test is "testConvTheo":
                params = {}
                params['filterSize'] = 7
                subTitle = ": I, I*M, IFT(FT(I).FT(M))"

            elif test is "testgaussianConvTheo":
                params = {}
                params['filterSize'] = 7
                subTitle = ": I, I*M, IFT(FT(I).FT(M))"

            elif test is "testmymask":
                params = {}
                params['filterSize'] = 7
                subTitle = ": I, IFT(FT(I).FT(M))"

            else:
                params = {}
                r,R = 5,None # for low-pass filter
                # 5,30 for band-pass filter
                # None, 30 for high-pass filter
                params['r'], params['R'] = r,R
                # let's assume r and R are not both None simultaneously
                if r is None:
                    filter="high pass" + " (R=" + str(R) + ")"
                elif R is None:
                    filter="low pass" + " (r=" + str(r) + ")"
                else:
                    filter="band pass" + " (r=" + str(r) + ", R=" + str(R) + ")"
                subTitle = ", " + filter + " filter"

            if test in testsUsingPIL:
                outs_pil = eval(test)(im_pil, params)
                outs_np = vpu.pil2np(outs_pil)
            else:
                # apply test to given image and given parameters
                outs_np = eval(test)(im, params)
            print("# images", len(outs_np))
            print(len(outs_np))

            vpu.showInGrid([im] + outs_np, title=nameTests[test] + subTitle)


if __name__ == "__main__":
    doTests()
