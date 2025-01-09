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
import visualPercepUtils as vpu

def FT(im):
    return fft.fftshift(fft.fft2(im))

def IFT(ft):
    return fft.ifft2(fft.ifftshift(ft))

def gaussianFilter(filterSize,sigma):
    gv1d = scipy.signal.windows.gaussian(filterSize, sigma).reshape(filterSize, 1)
    gv2d = np.outer(gv1d, gv1d)
    return gv2d/np.sum(gv2d)


def testgaussFT(im, params=None, sigma = None):
    print(sigma)
    filterSize = params['filterSize']
    filterMask = gaussianFilter(filterSize,sigma)  # the usually small mask
    filterBig = np.zeros_like(im, dtype=float)

    ## First, get sizes
    w, h = filterMask.shape
    w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2  # half width and height of the "big" mask

    ## Then, paste the small mask at the center using the sizes computed before as an aid
    filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterMask
    # FFT of the big filter
    filterBig = FT(fft.ifftshift(filterBig))
    phase = np.angle(filterBig)
    magnitude = np.absolute(filterBig)
    im2 = FT(im)
    im2 = im2 * filterBig
    im_filt_phase = np.angle(im2)
    im_filt_magnitude = np.absolute(im2)
    return [magnitude]


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
    tests = ['testgaussFT']
else:
    tests = ['testgaussFT']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testgaussFT': 'Gauss Filter Transform of Fourier'
             }

bSaveResultImgs = False

testsUsingPIL = []


def doTests(sigma):
    print("Testing on", files)
    for imfile in files:
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array
        for test in tests:
            if test is "testgaussFT":
                params = {}
                params['filterSize'] = 15
                #subTitle = ":|FT(M)|"

            if test in testsUsingPIL:
                outs_pil = eval(test)(im_pil, params, sigma)
                outs_np = vpu.pil2np(outs_pil)
            else:
                # apply test to given image and given parameters
                outs_np = eval(test)(im, params, sigma)
            print("# images", len(outs_np))
            print(len(outs_np))

            vpu.showInGrid(outs_np, title = f"Gauss Filter, σ={sigma}, Transform of Fourier: |FT(M)|")


if __name__ == "__main__":
    for i in [1,3,5,7,15,21,31,41,101]:
        doTests(i)